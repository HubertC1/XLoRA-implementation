import os
import argparse
from datetime import datetime
from human_eval.data import write_jsonl, read_problems
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# Argument parser for output directory
parser = argparse.ArgumentParser(description="Generate completions and save to JSONL")
parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
args = parser.parse_args()

# Get the current time and format it
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"{current_time}_completion.jsonl"
output_path = os.path.join(args.output_dir, output_filename)
#TODO: change model_path 
# checkpoint_path = "/work/u3076105/loraMoE/codingExpert/outputs/checkpoint-60"
checkpoint_path = "/work/u3076105/loraMoE/finetune/outputs/Magicoder-OSS/checkpoint-2350"
# checkpoint_path="unsloth/Meta-Llama-3.1-8B"

# Load the tokenizer and model from the checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
model = model.to("cuda")
model.eval()
print(model)

def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")

def generate_one_completion(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the prompt)
    generated_text = completion[len(prompt):].strip()
    
    return generated_text

problems = read_problems()

num_samples_per_task = 10
samples = []

# Outer progress bar for tasks
for task_id in tqdm(problems.keys(), desc="Processing tasks"):
    prompt = problems[task_id]["prompt"]
    
    # Inner progress bar for samples per task
    for _ in tqdm(range(num_samples_per_task), desc=f"Generating samples for task {task_id}", leave=False):
        completion = generate_one_completion(prompt)
        completion = fix_indents(completion)
        completion=filter_code(completion)
        samples.append(dict(task_id=task_id, completion=completion))

# Save the samples to the specified output path
write_jsonl(output_path, samples)
print(f"Completions saved to {output_path}")