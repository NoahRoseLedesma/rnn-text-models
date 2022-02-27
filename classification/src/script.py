import argparse
from data import load_data
from model import SentimentClassifier
import wandb

hyperparameter_defaults = {
    # Hyperparameters (training)
    "num_epochs": 10,
    "batch_size": 200,
    "learning_rate": 0.002725660987655802,

    # Hyperparameters (model),
    "embedding_size": 300,
    "hidden_size": 128,
    "lstm_layers": 3,
    "dropout": 0.6419267581924822,
    "bidirectional": True,
}

if __name__ == "__main__":
    wandb.init(config=hyperparameter_defaults, project="sentiment-analysis")

    print("Loading data...")
    data = load_data(wandb.config["batch_size"], wandb.config["embedding_size"])
    print("Data loaded.")

    print("Training model...")
    model = SentimentClassifier(wandb.config)
    model.train(data['train'])
    print("Model trained.")

    print("Evaluating model...")
    accuracy = model.evaluate(data['test'])
    print(f"Model evaluated. Test accuracy: {accuracy}")
    
    # Log the final accuracy to wandb
    wandb.log({"test_accuracy": accuracy})

