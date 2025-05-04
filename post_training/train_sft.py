import os
from trl import SFTTrainer, SFTTrainingArguments
from load_model import get_training_model_and_tokenizer
from dataset import load_sft_dataset


def main():
    model_name = "AI-MO/Kimina-Prover-Preview-Distill-1.5B"
    train_path = "data/sft_train.jsonl"
    output_dir = "outputs/sft"

    model, tokenizer = get_training_model_and_tokenizer(model_name)
    train_dataset = load_sft_dataset(train_path)

    training_args = SFTTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=100,
        save_steps=500,
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
