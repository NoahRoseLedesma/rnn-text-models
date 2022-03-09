from data import Dataset
from model import JokeGenerator
import torch
from transformers import logging
import sys
import json

with open("hyperparameters.json", 'r') as f:
    config = json.load(f)

if __name__ == "__main__":
    # Disable warning messages for the BERT transformer
    logging.set_verbosity_error()

    print("Loading data...")
    data = Dataset(
        "data/jokes.csv",
        config["sequence_length"]
    )
    print("Data loaded.")

    print("Loading the model...")
    model_state = torch.load("model.pt")
    vocab_size = model_state.fc.out_features
    model = JokeGenerator(config, vocab_size)
    model.load_state_dict(model_state.state_dict())
    model.BATCH_SIZE = 1

    # Read lines from stdin
    print("Ready.")
    print("> ", end="")
    sys.stdout.flush()
    for line in sys.stdin:
        print("Generating...")
        print(line, model.predict(data, line, 10))
        print("> ", end="")
        sys.stdout.flush()