from trl import DPOTrainer, DPOConfig
from load_model import get_training_model_and_tokenizer
from dataset import load_preference_dataset
from transformers import AutoModelForCausalLM


def main():
    model_name = "AI-MO/Kimina-Prover-Preview-Distill-1.5B"
    train_path = "data/dpo_prefs.jsonl"
    output_dir = "outputs/dpo"

    # Load trainable model and reference model
    model, tokenizer = get_training_model_and_tokenizer(model_name)
    reference_model = AutoModelForCausalLM.from_pretrained(model_name)
    reference_model.eval()

    train_dataset = load_preference_dataset(train_path)

    config = DPOConfig(
        learning_rate=1e-5,
        batch_size=1,
        microbatch_size=1,
        num_epochs=1,
        log_with="tensorboard",
    )

    trainer = DPOTrainer(
        model=model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=config,
        output_dir=output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()