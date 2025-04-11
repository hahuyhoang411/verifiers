import os
import verifiers as vf
from verifiers.parsers import XMLParser
from trl import GRPOConfig
from datasets import concatenate_datasets
from verifiers.tools.search import search_rag
from verifiers.utils import preprocess_dataset

"""
Multi-GPU training (single node, 4 training + 4 inference)

CUDA_VISIBLE_DEVICES=0,1 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-1.5B-Instruct' \
    --tensor_parallel_size 2 --max_model_len 4096 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num_processes 6 verifiers/examples/think_rag.py
"""

TOOL_PROMPT = """\
Answer questions confidently using your own knowledge. Always try to answer first. If you need more information, use a tool.

You have access to these tools:

{tool_descriptions}

Think step-by-step inside <think>...</think> tags:
- Break the question into smaller parts.
- Ask yourself:
  - Can I answer this with my knowledge?
  - Or do I need to use a tool?

If you need to use a tool, call it with JSON inside <tool>...</tool> tags:
- "name": tool name
- "args": arguments for the tool

Tool results will appear inside <result>...</result> tags. You can use tools multiple times if needed.

Always put your final answer inside <answer>...</answer> tags.

# Example usage:
## User: When was the first establishment that McDonaldization is named after?
## Assistant:
<think>
Subquestions:
- What is McDonaldization named after?
- When was the first establishment?

I know McDonaldization refers to McDonald's business practices, but I don't know the date.

So I need to search.
</think>
<tool>
{{"name": "search_rag", "args": {{"query": "first McDonald's establishment date", "num_results": 3}}}}
</tool>
## User:
<result>
"# Title: McDonald
# Context: The original **McDonald's** was opened by **Richard and Maurice McDonald** in **1940** in **San Bernardino, California**."
</result>
## Assistant:
<think>
Now I know: the first McDonald's was opened in 1940.

Let's verify:
- Does this answer the question about when the establishment that McDonaldization is named after was founded?
- Yes, McDonaldization refers to McDonald's business model, and the first McDonald's opened in 1940.

So my final answer is: 1940.
</think>
<answer>
1940
</answer>
"""

# Data
train_dataset = preprocess_dataset(name="tool_rl", split="train")
print(train_dataset)
print(train_dataset[0])

eval_dataset = preprocess_dataset(name="tool_rl", split="test")
print(eval_dataset)

vf_env = vf.ToolEnv(
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[search_rag],
    max_steps=5
)

think_parser = XMLParser(fields=["think", ("tool", "answer")])
vf_env.llm_parser = think_parser
vf_env.rubric.parser = think_parser

print(vf_env.system_prompt)

# Model
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "toolrl-rag_newrw_newer_" + model_name.split("/")[-1].lower()

training_args=GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=1e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=30,
    num_train_epochs=1,
    temperature=0.6,
    # max_steps=1000,
    bf16=True,
    max_grad_norm=0.1,
    num_iterations=2,
    beta=0.002,
    max_prompt_length=2048,
    max_completion_length=2048,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=20,
    num_generations=12,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=30,
    eval_accumulation_steps=1,
    eval_on_start=False,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    use_vllm=True,
    vllm_server_host="0.0.0.0", # replace with your inference server's host for multi-node setups
    vllm_server_port=8000,
    vllm_gpu_memory_utilization=0.9,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to="wandb",
    reward_weights=vf_env.get_reward_weights(),
    scale_rewards=False,
    epsilon_high=0.28,
    mask_truncated_completions=True,
    push_to_hub=True,
    hub_model_id="ToolMaestro-T1-1.5B"
)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=vf_env.get_reward_funcs(),
    env=vf_env,
    args=training_args,
    train_dataset=vf_env.get_dataset(),
    eval_dataset=vf_env.get_eval_dataset()
)
trainer.train() 