from data import Dataset
from model import JokeGenerator
import torch
from transformers import logging
import wandb
import json

with open("hyperparameters.json", 'r') as f:
    hyperparameter_defaults = json.load(f)

if __name__ == "__main__":
    wandb.init(config=hyperparameter_defaults, project="joke-generator")

    if not torch.cuda.is_available():
        print("Warning: No GPU found.")

    # Disable warning messages for the BERT transformer
    logging.set_verbosity_error()

    print("Loading data...")
    data = Dataset(
        "data/jokes.csv",
        wandb.config["sequence_length"]
    )
    print("Data loaded.")

    print("Training model...")
    model = JokeGenerator(wandb.config, len(data.vocab))
    model.train(data)
    print("Model trained.")

    print("Saving model...")
    # Save the model to a file
    torch.save(model, "model.pt")
    print("Model saved.")


    model.BATCH_SIZE = 1
    # Try some predictions
    print("\nSome sample predictions:")
    print(" ".join(model.predict(data, "What did one snowman say", 10)))

    print(" ".join(model.predict(data, "How many hipsters does it", 10)))

    print(" ".join(model.predict(data, "Which kitchen appliance tells the", 10)))