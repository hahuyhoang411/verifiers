import os
import verifiers as vf
from verifiers.parsers import XMLParser
from trl import GRPOConfig
from datasets import concatenate_datasets
from verifiers.tools.search import search_rag
from verifiers.utils import preprocess_dataset

"""
Multi-GPU training (single node, 2 training + 6 inference)
# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B or Qwen/Qwen2.5-1.5B-Instruct or Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=0,1 python verifiers/inference/vllm_serve.py --model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' \
    --tensor_parallel_size 2 --max_model_len 4096 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num_processes 6 verifiers/examples/think_rag.py
"""

TOOL_PROMPT = """\
You are a knowledgeable assistant. Answer questions confidently using your own knowledge. If you need more information, use a tool. Don't be biased toward using a tool unnecessarily.

You have access to these tools:

{tool_descriptions}

When reasoning, think step-by-step inside <think>...</think> tags:
- Break the question into smaller parts.
- MUST ask yourself before answering any questions: Can I answer this using my own knowledge?

If a tool is needed, call it using JSON inside <tool>...</tool> tags:
	• "name": the tool name
	• "args": the arguments required by the tool

Tool results will appear inside <result>...</result> tags. You can call tools multiple times if the search results don't contain context to answer the question.

Always put your final answer inside <answer>...</answer> tags.

⸻

# Example 1: Answerable without a tool

## User: What is McDonaldization named after?
## Assistant:
<think>
Let me break this down by identifying the key part of the question:
- The question asks what McDonaldization is named after.
This means I need to recall the origin or inspiration behind the term "McDonaldization".

• Can I answer this using my knowledge?
Yes.

I know that McDonaldization is named after the McDonald's fast-food chain. The term was introduced by sociologist George Ritzer in his 1993 book *The McDonaldization of Society*.
</think>
<answer>
McDonald's fast-food chain
</answer>

⸻

# Example 2: Requires a tool

## User: When was the first McDonald's established?
## Assistant:
<think>
Let me analyze the question:
- It's asking when the first McDonald's was established.
- This is a factual, date-based question about a historical event.
I don't recall the exact year, so I'll need to use a tool to retrieve it.

• Can I answer this using my knowledge?
No. I need to use a search tool.

I don't know the exact year offhand, so I'll search for it.
</think>
<tool>
{{"name": "search_rag", "args": {{"query": "first McDonald's establishment date", "num_results": 3}}}}
</tool>
## User: This context is onlu returned when the tool is used
<result>
"Title: McDonald  
Context: The original McDonald's was opened by Richard and Maurice McDonald in 1940 in San Bernardino, California."
</result>
## Assistant:
<think>
Based on the result, I now know that the first McDonald's was opened in 1940.
</think>
<answer>
1940
</answer>
"""

# Data
train_dataset = preprocess_dataset(name="tool_rl", split="train")
# train_dataset = train_dataset.select(range(1000))
print(train_dataset)
print(train_dataset[0])

eval_dataset = preprocess_dataset(name="tool_rl", split="test")
# eval_dataset = eval_dataset.select(range(100))
print(eval_dataset)

vf_env = vf.ToolEnv(
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[search_rag],
    max_steps=5
)

# print(vf_env.system_prompt)

# Model
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # "Qwen/Qwen2.5-7B-Instruct"
model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "ToolMaestro-T1-Ins-7B-v0.1-researchdata" + model_name.split("/")[-1].lower()

training_args=GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=3e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=30,
    # num_train_epochs=1,
    temperature=0.6,
    max_steps=300, # 1 epoch = 139 steps
    bf16=True,
    max_grad_norm=0.1,
    num_iterations=4,
    beta=0.01,
    max_prompt_length=2048,
    max_completion_length=2048,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_generations=12,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=50,
    eval_accumulation_steps=1,
    eval_on_start=True,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    use_vllm=True,
    vllm_server_host="0.0.0.0",
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
    hub_model_id="ToolMaestro-T1-7B-Ins-v0.1",
    # use_liger_loss=True,
    loss_type="dr_grpo"
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