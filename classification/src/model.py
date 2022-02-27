# Define the model
import torch
from torch import nn
import numpy as np
import wandb

# This model classifies reviews as positive or negative
# The inputs are word embeddings for each word in the review
class SentimentClassifier(nn.Module):
    # Training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, config):
        super().__init__()

        # Hyperparameters (training)
        self.NUM_EPOCHS = config["num_epochs"]
        self.BATCH_SIZE = config["batch_size"]
        self.LEARNING_RATE = config["learning_rate"]
        
        # Hyperparameters (model)
        self.EMBEDDING_SIZE = config["embedding_size"]
        self.HIDDEN_SIZE = config["hidden_size"]
        self.LSTM_LAYERS = config["lstm_layers"]
        self.DROPOUT = config["dropout"]
        self.BIDIRECTIONAL = config["bidirectional"]
        
        # Define the model architecture
        self.rnn = nn.LSTM(
            input_size=self.EMBEDDING_SIZE,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.LSTM_LAYERS,
            dropout=self.DROPOUT,
            bidirectional=self.BIDIRECTIONAL,
            batch_first=True)

        self.fc = nn.Linear(self.HIDDEN_SIZE * 2, 1)

        self.to(self.device)
    
    # Define the model's forward pass
    def forward(self, x):
        # Pass the input through the RNN
        output, (hidden, _) = self.rnn(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # Pass the output of the RNN through a fully connected layer
        return self.fc(hidden)
    
    # Define the model's training function
    def train(self, batch_iterator):
        # Define the loss function
        loss_fn = nn.BCEWithLogitsLoss().to(self.device)

        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), self.LEARNING_RATE)

        # Train the model
        for epoch in range(self.NUM_EPOCHS):
            epoch_loss = 0
            epoch_accuracy = 0

            for i, (X, y) in enumerate(batch_iterator):
                # Zero the gradient
                optimizer.zero_grad()
                # Perform a forward pass
                X = torch.FloatTensor(X).to(self.device)
                y = torch.FloatTensor(y).to(self.device)
                y_pred = self.forward(X).squeeze(1)
                # Calculate the loss
                loss = loss_fn(y_pred, y)
                # Backpropagate the loss
                loss.backward()
                # Update the parameters
                optimizer.step()
                # Contribute to the epoch loss and accuracy
                epoch_loss += loss.item()
                epoch_accuracy += self.get_accuracy(y_pred, y)

                # Free up memory
                del X, y, loss
                # Clear cache
                torch.cuda.empty_cache()

            # Reset the batch iterator
            batch_iterator.reset()
            
            # Average the epoch loss and accuracy
            epoch_loss /= len(batch_iterator)
            epoch_accuracy /= len(batch_iterator)

            if epoch % 1 == 0:
                # Print the loss
                print(f"Epoch {epoch + 1}/{self.NUM_EPOCHS}, Loss: {epoch_loss:.4f}")
                # Print the accuracy
                print(f"Accuracy: {epoch_accuracy:.4f}")
            
            # Log the loss and accuracy to Weights and Biases
            wandb.log({"training_loss": epoch_loss})
            wandb.log({"training_accuracy": epoch_accuracy})
        
    # Inference function
    def predict(self, X):
        # Do the forward pass in batches
        y_pred = []
        for i in range(0, len(X), self.BATCH_SIZE):
            x_batch = torch.FloatTensor(X[i:i+self.BATCH_SIZE]).to(self.device)
            y_pred.append(torch.sigmoid(self.forward(x_batch)).cpu().detach().numpy())
            del x_batch
        
        # Concatenate the results
        y_pred = np.concatenate(y_pred, axis=0)
        return y_pred.round().squeeze()
    
    # Evaluate the model and return the accuracy
    def evaluate(self, batch_iterator):
        accuracy = 0
        # Do the forward pass in batches
        for (X, y) in batch_iterator:
            X = torch.FloatTensor(X).to(self.device)
            y = torch.FloatTensor(y).to(self.device)
            y_pred = self.forward(X).squeeze(1)
            accuracy += self.get_accuracy(y_pred, y)
            del X, y
        batch_iterator.reset()
        
        return accuracy / len(batch_iterator)

    def get_accuracy(self, y_pred, y):
        y_pred = torch.round(torch.sigmoid(y_pred))
        correct = (y_pred == y).float()
        return correct.sum() / len(correct)

