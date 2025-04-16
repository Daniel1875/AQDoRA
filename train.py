

from typing import overload
import gc
import os

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch import profiler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

hf_token = None
with open('FULL_ACCESS.txt', 'r') as file:
    hf_token = file.readline()
assert torch.cuda.is_available()

device = 'cuda'
# torch.set_default_device(device)

model = AutoModelForCausalLM.from_pretrained('ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf', token=hf_token, device_map='auto', torch_dtype='auto', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained('ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf', token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

torch.cuda.memory._record_memory_history()

for param in model.parameters():
    param.requires_grad = False
lora_config = LoraConfig(
    r=8,
    target_modules=['q_proj', 'k_proj', 'o_proj'],
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

torch.cuda.empty_cache()
gc.collect()

print(model)
model.print_trainable_parameters()

# PART 3. TRAINERS
training_args = TrainingArguments(
    output_dir='results',
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    # torch_empty_cache_steps=1,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5' # 7.5 for RTX-6000, 8.0 for A100
data = load_dataset('wikitext', 'wikitext-2-raw-v1')
data = data.map(lambda samples: tokenizer(samples['text']), batched=True)

# data = load_dataset('Abirate/english_quotes')
# data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)

class MemoryUsageCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Print memory usage
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        torch.cuda.empty_cache()
        print(f"Allocated memory: {allocated_memory:.2f} MB")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data['train'],
    eval_dataset=data['validation'],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[MemoryUsageCallback(),]
)

torch.cuda.memory._dump_snapshot('my_snapshot.pickle')
torch.cuda.empty_cache()
gc.collect()

# PART 4. TRAINING AND EVALUATION
trainer.train()
gc.collect()

model.save_pretrained('Mixtral-8x7b/AQLMLoRA', from_pt=True)

