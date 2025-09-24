# Make it more memory efficient by monkey patching the LLaMA model with xformers attention.

# Need to call this before importing transformers.
from model_zoo.Cambrian.train.llama_xformers_attn_monkey_patch import (
    replace_llama_attn_with_xformers_attn,
)

replace_llama_attn_with_xformers_attn()

from model_zoo.Cambrian.train.train_fsdp import train

if __name__ == "__main__":
    train()
