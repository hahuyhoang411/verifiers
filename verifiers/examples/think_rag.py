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
Answer questions confidently using your knowledge. Try to answer the question first. If more information is needed, use the search tool. Strictly follow the instructions.

You have access to the following tools to help solve problems:

{tool_descriptions}

Think step-by-step inside <think>...</think> tags to decide:
- Can I answer using my knowledge?
- Or do I need to use the search tool?

If you use a tool, call it with JSON inside <tool>...</tool> tags:
- "name": the name of the tool to use
- "args": the arguments for the tool

Tool output will appear inside <result>...</result> tags. You can use a tool more than once if needed.

Put your final answer inside <answer>...</answer> tags.

# Example usage 1:
Question: When was the first establishment that McDonaldization is named after, open in the country Horndean is located?
<think>
I don't know what McDonaldization is named after. I need to use the search tool to find it.
</think>
<tool>
{{"name": "search_rag", "args": {{"query": "McDonaldization named after", "num_results": 3}}}}
</tool>
<think>
Now I know McDonaldization is named after McDonald's.

And I know Horndean is located in England. So I don't need to use the search tool for that.

But I don't know when the first McDonald's opened in England. I need to use the search tool for that.
</think>
<tool>
{{"name": "search_rag", "args": {{"query": "first McDonald's opening date England", "num_results": 3}}}}
</tool>
<think>
Now I know that the first McDonald's restaurant in England, which is what McDonaldization is named after, opened in 1974.
</think>
<answer>
1974
</answer>

If the answer is not found in the searched context, issue a new query with tool.
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