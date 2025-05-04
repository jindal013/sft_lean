import os
import random
import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(csv_path: str, tokenizer, test_size: float, seed: int):
    # load csv and tokenize
    ds = load_dataset("csv", data_files={"train": csv_path})["train"]
    
    def tokenize_example(example):
        messages = [
            {"role": "user", "content": f"{example['instruction']} {example['input']}"},
            {"role": "assistant", "content": example['output']},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized = tokenizer(prompt, return_tensors="pt")
        return {"prompt": prompt, **tokenized}

    ds = ds.map(tokenize_example, batched=False)
    ds = ds.shuffle(seed=seed)
    splits = ds.train_test_split(test_size=test_size)
    return splits['train'], splits['test']


def main(args):
    # set random seed for reproducibility
    set_seed(args.seed)

    # load tokenizer and model with lora
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="right", add_eos_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=getattr(torch, args.precision),
        device_map="auto"
    )
    model.gradient_checkpointing_enable()

    # identify linear modules for lora
    linear_module_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    target_modules = linear_module_names[: args.modules_limit]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # prepare datasets
    train_ds, eval_ds = prepare_dataset(args.data_path, tokenizer, args.test_size, args.seed)

    # training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        optim=args.optim,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to=args.report_to or []
    )

    # initialize sft trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        dataset_text_field="prompt",
        args=training_args
    )

    # start training
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with LoRA and TRL SFTTrainer")
    parser.add_argument("--model_id", type=str, required=True, help="KAMINO MODEL HERE") # TODO for me: add model here!!
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset with instruction,input,output columns")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of data for evaluation split")
    parser.add_argument("--modules_limit", type=int, default=10, help="Max number of linear modules to apply LoRA")
    parser.add_argument("--lora_rank", type=int, default=2, help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=float, default=0.5, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--precision", type=str, choices=["bfloat16", "float32", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--report_to", nargs="*", default=None)
    args = parser.parse_args()
    main(args)
