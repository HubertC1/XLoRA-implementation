from unsloth import FastLanguageModel
from datasets import load_dataset, concatenate_datasets
from instructions import instructions_dict
import random
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, XLoraConfig
import xlora

import torch

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    use_cache = False,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model_config = AutoConfig.from_pretrained("unsloth/Meta-Llama-3.1-8B")
config = XLoraConfig(
    task_type="CAUSAL_LM",
    hidden_size=model_config.hidden_size,
    xlora_depth=4,
    layerwise_scalings = True,
    adapters={
        "codine_expert0": "/work/u3076105/loraMoE/finetune/outputs/vault_func/checkpoint-6500",
        "coding_expert1": "/work/u3076105/loraMoE/finetune/outputs/Magicoder-OSS/checkpoint-2350",
        "coding_expert2": "/work/u3076105/loraMoE/finetune/outputs/coding-prompts-small /checkpoint-500"
    },
)
model = prepare_model_for_kbit_training(model)
# xlora_model= xlora.add_xlora_to_model(
#     model=model,
#     xlora_config=xlora.xLoRAConfig(
#         model_config.hidden_size,
#         base_model_id="unsloth/Meta-Llama-3.1-8B",
#         xlora_depth=4,
#         layerwise_scalings = True,
#         # Remove or set device to None to let Accelerate handle it
#         adapters={
#             "codine_expert0": "/work/u3076105/loraMoE/finetune/outputs/vault_func/checkpoint-6500",
#             "coding_expert1": "/work/u3076105/loraMoE/finetune/outputs/Magicoder-OSS/checkpoint-2350",
#             "coding_expert2": "/work/u3076105/loraMoE/finetune/outputs/coding-prompts-small /checkpoint-500"
#         },
#         device="cuda:0"
#     ),
#     verbose=True,
# )



xlora_model = get_peft_model(model, config)

print(xlora_model)

tokenizer.pad_token = tokenizer.eos_token
#Finetuning ================= ================= ================= ================= =================

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
def formatting_prompts_func(examples, instruction, input ,output):
    # instructions = examples["prompt"]
    inputs      = examples[input]
    outputs      = examples[output]
    if instruction == "none":
        texts = []
        i = 0
        for input, output in zip(inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            # full_input = "context:" + context + "\n" + "question:" + question + "answer:"
            text = alpaca_prompt.format(random.choice(instructions), input, output) + EOS_TOKEN
            texts.append(text)
        # print(texts)
        return { "text" : texts, }
    else:
        instructs = examples[instruction]
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



# dataset = dataset['train_medium'].map(formatting_prompts_func, batched = True,)
# dataset = load_dataset("camel-ai/physics", split=["train"])
# dataset = load_dataset("meta-math/MetaMathQA", split=["train"])
# dataset = load_dataset("rajpurkar/squad", split=["train"])
dataset1 = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split=["train"])[0].select(range(20000))
dataset1 = dataset1.map(lambda examples : formatting_prompts_func(examples, instruction = "none", input = "problem", output = "solution"), batched=True,)
dataset2 = load_dataset("Fsoft-AIC/the-vault-function", split_set=["train/medium"], languages=['python'])['train_medium'].select(range(20000))
dataset2 = dataset2.map(lambda examples : formatting_prompts_func(examples, instruction = "none", input = "docstring", output = "code"), batched = True,)
dataset3 = load_dataset("perlthoughts/coding-prompts-small")["train"].select(range(20000))
dataset3 = dataset3.map(lambda examples : formatting_prompts_func(examples, instruction = "instruction", input = "input", output = "output"), batched = True,)
# print(dataset["text"])
# Step 2: Concatenate datasets
combined_dataset = concatenate_datasets([dataset1, dataset2, dataset3])

# Step 3: Shuffle the combined dataset
shuffled_dataset = combined_dataset.shuffle(seed=42)

shuffled_dataset.save_to_disk("/work/u3076105/loraMoE/dataset/code123Truncate")


trainer = SFTTrainer(
    model = xlora_model,
    tokenizer = tokenizer,
    train_dataset = shuffled_dataset,
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
        save_steps = 30,
        # fp16 = not is_bfloat16_supported(),
        # bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_bnb_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs/xlora_layerwise",
    ),
)

trainer.optimizer = None
trainer.lr_scheduler = None

trainer_stats = trainer.train("/work/u3076105/loraMoE/finetune/outputs/XLoRA_3code/checkpoint-30")

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