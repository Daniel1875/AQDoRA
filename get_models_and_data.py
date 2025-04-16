
from typing import overload

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import aqlm

hf_token = None
with open('FULL_ACCESS.txt', 'r') as file:
    hf_token = file.readline()

device = 'cuda'
torch.set_default_device(device)

# PART 1. MODEL

# Mixtral 8x7B with AQLM
mixtralaqlm = AutoModelForCausalLM.from_pretrained('ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf', token=hf_token, device_map='auto', torch_dtype='auto', low_cpu_mem_usage=True)
mixtralaqlm.config.pad_token_id = mixtralaqlm.config.eos_token_id

# PART 2. DATA
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
print('This actually finished running.')