

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
from datasets import load_dataset

hf_token = None
with open('FULL_ACCESS.txt', 'r') as file:
    hf_token = file.readline()
assert torch.cuda.is_available()

device = 'cuda'
torch.set_default_device(device)

model = AutoModelForCausalLM.from_pretrained('ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf', token=hf_token, device_map='auto', torch_dtype='auto', low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(model, 'Mixtral-8x7b/AQLMLoRA')
tokenizer = AutoTokenizer.from_pretrained('ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf', token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

print(model)
model.print_trainable_parameters()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5'
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
    
    perplexity = torch.exp(loss)
    return perplexity.item()

total_perplexity = 0
num_samples = 0

for i in range(0, len(dataset), 8):
    batch = dataset[i:i + 8]
    texts = batch['text']
    
    for text in texts:
        perplexity = calculate_perplexity(model, tokenizer, text)
        print(perplexity)
        total_perplexity += perplexity
        num_samples += 1

average_perplexity = total_perplexity / num_samples if num_samples > 0 else float('inf')

print(f'Average Perplexity on WikiText-2: {average_perplexity:.2f}')