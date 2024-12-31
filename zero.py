import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import deepspeed
import time
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import subprocess
import re

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

def get_max_memory_usage(gpus):
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    memory_usages = [int(x) for x in result.stdout.decode().strip().split('\n')]
    usage = {}
    for gpu in gpus:
        usage[gpu] = memory_usages[gpu]
    return usage


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
 
    model.seqlen = model.config.max_position_embeddings  # 2048
    return model

model= get_llm("/data2/LLM/Qwen2.5/Qwen-3B")
tokenizer = AutoTokenizer.from_pretrained("/data2/LLM/Qwen2.5/Qwen-3B")

ds_config = {
    "fp16": {"enabled": True},
    "bf16": {"enabled": False},
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu"
        }
    },
    "train_micro_batch_size_per_gpu": 1
}

ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()

start = time.time()
inputs = tokenizer.encode("DeepSpeed is", return_tensors='pt').to(f"cuda:{local_rank}")
outputs = model.generate(inputs, max_new_tokens=20)
print(get_max_memory_usage([0]))
output_str = tokenizer.decode(outputs[0])
end = time.time()
print(output_str)
print("ZeRO-Inference time:", end-start)
