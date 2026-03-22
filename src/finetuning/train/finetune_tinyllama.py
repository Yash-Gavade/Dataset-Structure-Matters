import argparse
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["alpaca", "dolly", "oasst1"]
    )
    parser.add_argument("--data_dir", type=str, default="finetuning_data")
    parser.add_argument("--output_root", type=str, default="finetuned_models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_train_samples", type=int, default=1000)
    return parser.parse_args()


def tokenize_function(examples, tokenizer, max_seq_length):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    args = parse_args()

    dataset_path = os.path.join(args.data_dir, args.dataset_name)
    output_dir = os.path.join(args.output_root, f"tinyllama_{args.dataset_name}")

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    # Quick experiment: limit samples for faster training
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    print(f"Training samples: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    model.config.use_cache = False

    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    fp16_flag = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="none",
        fp16=fp16_flag,
        bf16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Done. Saved model to: {output_dir}")


if __name__ == "__main__":
    main()