import warnings
from typing import Callable, Optional, Union, Any, List
import logging
import json

import numpy as np 
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
from peft import PeftConfig # type: ignore
import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available
)
from verifiers import RewardFunc
from verifiers.envs.environment import Environment
from verifiers.utils.logging_utils import print_prompt_completions_sample
from verifiers.imports import LLM, SamplingParams
from verifiers.inference.vllm_client import VLLMClient

# monkey patch vllm client
import trl.extras.vllm_client
trl.extras.vllm_client.VLLMClient = VLLMClient

from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad

if is_wandb_available():
    import wandb


logger = logging.getLogger(__name__)

# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    # Handle case where count <= 1 to avoid division by zero or negative denominator
    if count <= 1:
        return torch.tensor(0.0, device=tensor.device)
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

def safe_mean(arr):
    """Calculate mean safely, returning 0.0 if array is empty."""
    return np.mean(arr) if len(arr) > 0 else 0.0

class GRPOEnvTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            scale_rewards: bool = False,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):
        self.vllm_client = None
        if not args.use_vllm: # type: ignore
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if not (callable(reward_funcs) or (isinstance(reward_funcs, list) and all(callable(f) for f in reward_funcs))):
            raise ValueError("reward_funcs must be a function or a list of functions. Use vLLM to host neural reward models.")

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        self.env = env
        self.scale_rewards = scale_rewards
        self.sampling_params = SamplingParams(
            max_tokens=self.max_completion_length,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=-1 if self.top_k is None else self.top_k,
            min_p=0.0 if self.min_p is None else self.min_p,
            repetition_penalty=self.repetition_penalty
        )
        # Store the index for the correctness reward function for efficiency
        self._correctness_reward_idx = -1
        try:
            # Attempt to find the specific function instance if available
            # This might require the user to ensure the function object is accessible
            # or rely on a naming convention. Let's use a name convention for robustness.
            correctness_func_name = "correct_answer_reward_func"
            for i, func in enumerate(self.reward_funcs):
                 if hasattr(func, "__name__") and func.__name__ == correctness_func_name:
                     self._correctness_reward_idx = i
                     break
            if self._correctness_reward_idx == -1:
                 logger.warning(f"Could not find reward function named '{correctness_func_name}'. Correctness metrics might not be logged.")
        except Exception as e:
             logger.warning(f"Error finding correctness reward function index: {e}. Correctness metrics might not be logged.")


    def _generate_and_score_completions(
         self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs] # type: ignore
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs] # type: ignore
        research_required_local = [x.get("research", False) for x in inputs] # Default to False if missing

        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False # type: ignore
        ) # type: ignore
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs) # type: ignore
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather the original prompts in message dict form, not the text form
        all_prompts = gather_object(prompts)
        if self.accelerator.is_main_process:
            env_result = self.env.generate(
                prompts=all_prompts,
                llm=self.vllm_client, # type: ignore
                sampling_params=self.sampling_params,
            )
            completion_ids = env_result['ids']
            completion_messages = env_result['messages']
            completion_mask = env_result['mask']

        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]

        tool_used_local = []
        if hasattr(self.env, 'llm_parser'): # Check if parser exists
            for trajectory in completion_messages:
                used_tool_in_traj = False
                if isinstance(trajectory, list): # Should be a list of dicts
                     for msg in trajectory:
                         if msg.get("role") == "assistant":
                             try:
                                 parsed = self.env.llm_parser.parse(msg.get("content", ""))
                                 if hasattr(parsed, 'tool') and parsed.tool is not None:
                                     used_tool_in_traj = True
                                     break # Found tool use, no need to check further in this trajectory
                             except Exception:
                                 # Ignore parsing errors for this check
                                 pass
                tool_used_local.append(used_tool_in_traj)
        else:
            logger.warning("Environment does not have 'llm_parser'. Cannot track tool usage.")
            tool_used_local = [False] * len(completion_messages) # Assume no tool use if no parser


        # Pad + mask after per-sequence EOS tokens
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id) # type: ignore

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=0)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) # (B, P+C)

        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # use message dicts for reward function inputs
        completions = completion_messages
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]] # type: ignore
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys} # type: ignore
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs) # type: ignore

            output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()} # type: ignore
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx] # type: ignore
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        is_correct_local = []
        if self._correctness_reward_idx != -1:
             # Assuming correctness is 1.0 for correct, 0.0 for incorrect
             correctness_tensor = rewards_per_func[:, self._correctness_reward_idx]
             # Handle potential NaNs if the correctness function didn't apply
             is_correct_local = (~torch.isnan(correctness_tensor) & (correctness_tensor == 1.0)).tolist()
        else:
             is_correct_local = [False] * len(prompts) # Default if correctness func not found

        rewards_per_func_gathered = gather(rewards_per_func) # <-- RENAME gathered version

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func_gathered * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # type: ignore

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        advantages = (rewards - mean_grouped_rewards)

        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # type: ignore
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        if self.scale_rewards:
            # Scale the rewards to be between 0 and 1
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        # process_slice defined earlier is correct here
        advantages = advantages[process_slice]

        all_tool_used = gather_object(tool_used_local)
        all_is_correct = gather_object(is_correct_local)
        all_research_required = gather_object(research_required_local)


        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item() # type: ignore
        self._metrics[mode]["completion_length"].append(completion_length)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__ # type: ignore
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            # Use the GATHERED rewards_per_func here
            mean_rewards = torch.nanmean(rewards_per_func_gathered[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
            # Use the GATHERED rewards_per_func here
            std_rewards = nanstd(rewards_per_func_gathered[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item()) # type: ignore


        if self.accelerator.is_main_process:
            # Convert gathered lists to numpy arrays for easier boolean indexing
            all_tool_used_np = np.array(all_tool_used)
            all_is_correct_np = np.array(all_is_correct)
            all_research_required_np = np.array(all_research_required)
            total_generations = len(all_tool_used_np)

            if total_generations > 0:
                # Tool Usage Rates
                tool_usage_rate = safe_mean(all_tool_used_np)
                self._metrics[mode]["metrics/tool_usage_rate"].append(tool_usage_rate)

                required_mask = all_research_required_np == True
                not_required_mask = all_research_required_np == False

                tool_usage_rate_required = safe_mean(all_tool_used_np[required_mask])
                self._metrics[mode]["metrics/tool_usage_rate_required"].append(tool_usage_rate_required)

                tool_usage_rate_not_required = safe_mean(all_tool_used_np[not_required_mask])
                self._metrics[mode]["metrics/tool_usage_rate_not_required"].append(tool_usage_rate_not_required)

                # Correctness Conditional on Tool Use
                tool_used_mask = all_tool_used_np == True
                tool_not_used_mask = all_tool_used_np == False

                accuracy_tool_used = safe_mean(all_is_correct_np[tool_used_mask])
                self._metrics[mode]["metrics/accuracy_tool_used"].append(accuracy_tool_used)

                accuracy_tool_not_used = safe_mean(all_is_correct_np[tool_not_used_mask])
                self._metrics[mode]["metrics/accuracy_tool_not_used"].append(accuracy_tool_not_used)

                # Correctness Conditional on Tool Use AND Requirement
                accuracy_tool_used_required = safe_mean(all_is_correct_np[tool_used_mask & required_mask])
                self._metrics[mode]["metrics/accuracy_tool_used_required"].append(accuracy_tool_used_required)

                accuracy_tool_not_used_required = safe_mean(all_is_correct_np[tool_not_used_mask & required_mask])
                self._metrics[mode]["metrics/accuracy_tool_not_used_required"].append(accuracy_tool_not_used_required)

                accuracy_tool_used_not_required = safe_mean(all_is_correct_np[tool_used_mask & not_required_mask])
                self._metrics[mode]["metrics/accuracy_tool_used_not_required"].append(accuracy_tool_used_not_required)

                accuracy_tool_not_used_not_required = safe_mean(all_is_correct_np[tool_not_used_mask & not_required_mask])
                self._metrics[mode]["metrics/accuracy_tool_not_used_not_required"].append(accuracy_tool_not_used_not_required)


        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts)
            completions_to_log = gather_object(completions)
            rewards_to_log = rewards.tolist() # Use the gathered rewards

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        # Log only the user part of the first prompt for brevity
                        [str(prompts_to_log[0][-1]["content"]) if prompts_to_log and prompts_to_log[0] else "N/A"],
                        # Log only the first completion message list
                        [completions_to_log[0] if completions_to_log else []],
                        # Log only the first reward
                        [rewards_to_log[0] if rewards_to_log else 0.0],
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None: # type: ignore
                    import pandas as pd

                    # Log a sample to W&B table (e.g., first N examples)
                    log_limit = min(5, len(prompts_to_log)) # Limit logged samples
                    table_data = {
                        "step": [str(self.state.global_step)] * log_limit,
                        # Extract final user prompt content for simplicity
                        "prompt": [str(p[-1]['content']) if p else 'N/A' for p in prompts_to_log[:log_limit]],
                        # Convert message list to string representation
                        "completion": [json.dumps(c) if c else 'N/A' for c in completions_to_log[:log_limit]],
                        "reward": rewards_to_log[:log_limit],
                        "tool_used": all_tool_used[:log_limit],
                        "correct": all_is_correct[:log_limit],
                        "research_required": all_research_required[:log_limit],
                    }
                    try:
                        df = pd.DataFrame(table_data)
                        wandb.log({"completions_sample": wandb.Table(dataframe=df)}) # type: ignore
                    except Exception as e:
                        logger.error(f"Error creating or logging W&B table: {e}")


        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }