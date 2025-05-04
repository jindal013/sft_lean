from trl import PPOTrainer, PPOConfig
from load_model import get_training_model_and_tokenizer
from dataset import load_gpro_dataset


def reward_function(samples, **kwargs):
    """
    this still needs to be defined lol
    """
    rewards = []
    for text in samples:
        # p]laceholder: assign +1 for containing proof keyword, -1 otherwise
        rewards.append(1.0 if "theorem" in text else -1.0)
    return rewards


def main():
    model_name = "AI-MO/Kimina-Prover-Preview-Distill-1.5B"
    train_path = "data/gpro_train.jsonl"
    output_dir = "outputs/gpro"

    model, tokenizer = get_training_model_and_tokenizer(model_name)
    train_dataset = load_gpro_dataset(train_path)

    config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=1,
        forward_batch_size=1,
        gradient_accumulation_steps=1,
        log_with="tensorboard",
    )

    ppo_trainer = PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        **config
    )

    for example in train_dataset:
        query = example["query"]
        response = ppo_trainer.generate(query)
        reward = reward_function(response)
        ppo_trainer.step(
            queries=[query],
            responses=response,
            rewards=reward,
        )

    ppo_trainer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
