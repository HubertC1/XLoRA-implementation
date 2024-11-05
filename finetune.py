from unsloth import FastLanguageModel
from datasets import load_dataset
from instructions import instructions_dict
import random


import torch

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",          # Phi-3 2x faster!d
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# model.eval()
# print("before fast type:", type(model))
# print("model:", model)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# instructions = instructions_dict['math']
instructions = instructions_dict['coding']

# model.eval()
# print("type", type(model))
# print("targetmodules:", model.target_modules)
# print(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    # instructions = examples["prompt"]
    instructs = examples["instruction"]
    inputs      = examples["input"]
    # contexts = examples["context"]
    # questions = examples["question"]
    outputs      = examples["output"]
    texts = []
    i = 0
    for instruct, input, output in zip(instructs, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        # full_input = "context:" + context + "\n" + "question:" + question + "answer:"
        text = alpaca_prompt.format(random.choice(instructions), instruct + "\n Input:" + input, output) + EOS_TOKEN
        texts.append(text)
    # print(texts)
    return { "text" : texts, }
pass


# dataset = load_dataset("openai/openai_humaneval", split = "train")
# dataset = load_dataset("Fsoft-AIC/the-vault-function", split_set=["train/medium"], languages=['python'])
# dataset = dataset['train_medium'].map(formatting_prompts_func, batched = True,)

dataset = load_dataset("perlthoughts/coding-prompts-small")
dataset = dataset["train"].map(formatting_prompts_func, batched = True,)
# dataset = dataset['train_medium'].map(formatting_prompts_func, batched = True,)
# dataset = load_dataset("camel-ai/physics", split=["train"])
# dataset = load_dataset("meta-math/MetaMathQA", split=["train"])
# dataset = load_dataset("rajpurkar/squad", split=["train"])
# dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split=["train"])
# processed_dataset = dataset[0].map(formatting_prompts_func, batched=True,)
# print(dataset["text"])

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 3020,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
       output_dir = "outputs/coding-prompts-small ",
    ),
)

trainer_stats = trainer.train()

# alpaca_prompt = Copied from above
# FastLanguageModel.for_inference(model) # Enable native 2x faster inference
# inputs = tokenizer(
# [
#     alpaca_prompt.format(
#         "Solve the following physics problem", # instruction
#         "I want to calculate the 10th fibonacci number with python.", # input
#         "", # output - leave this blank for generation!
#     )
# ], return_tensors = "pt").to("cuda")

# outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
# decoded_output = tokenizer.batch_decode(outputs)
# print(decoded_output)