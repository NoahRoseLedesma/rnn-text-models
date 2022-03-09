import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
import wandb

class JokeGenerator(nn.Module):
    # Training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, config, vocab_size):
        super().__init__()

        self.vocab_size = vocab_size

        # Hyperparameters (training)
        self.NUM_EPOCHS = config["num_epochs"]
        self.BATCH_SIZE = config["batch_size"]
        self.LEARNING_RATE = config["learning_rate"]
        
        # Hyperparameters (model)
        self.EMBEDDING_SIZE = config["embedding_size"]
        self.HIDDEN_SIZE = config["hidden_size"]
        self.GRU_LAYERS = config["gru_layers"]
        self.DROPOUT = config["dropout"]
        self.BIDIRECTIONAL = config["bidirectional"]
        self.SEQUENCE_LENGTH = config["sequence_length"]
        self.FINE_TUNE_EMBEDDINGS = config["fine_tune_embeddings"]

        # Bidirectional constant
        self.D = 2 if self.BIDIRECTIONAL else 1
        
        self.embeddings = nn.Embedding(self.vocab_size, self.EMBEDDING_SIZE)

        # Define the model architecture
        self.rnn = nn.GRU(
            input_size=self.EMBEDDING_SIZE,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.GRU_LAYERS,
            dropout=self.DROPOUT,
            bidirectional=self.BIDIRECTIONAL,
            batch_first=True
        )

        self.fc = nn.Linear(self.HIDDEN_SIZE * self.D, vocab_size)

        self.to(self.device)
    
    # Define the model's initial state
    def get_init_state(self):
        return torch.zeros(self.GRU_LAYERS * self.D, self.BATCH_SIZE, self.HIDDEN_SIZE).to(self.device)

    # Define the model's forward pass as a stateful function.
    # It recieves the previous state and the current input.
    def forward(self, x, last_state):
        # Embed the input
        x = self.embeddings(x)
        # Pass the input through the RNN
        output, next_state = self.rnn(x, last_state)
        # Pass the output of the RNN through a fully connected layer
        output = self.fc(output)
        # Take the output from the last time step
        output = output[:, -1, :].squeeze(1)
        # Apply a log softmax to the output
        output = torch.nn.functional.log_softmax(output, dim=1)

        return output, next_state
    
    # Define the model's training function
    def train(self, dataset):
        # Define the loss function
        loss_fn = nn.NLLLoss()

        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), self.LEARNING_RATE)

        # Create a dataloader for the training dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.BATCH_SIZE, drop_last=True)

        # Initialize the embedding weights
        self.embeddings.weight = torch.nn.Parameter(torch.FloatTensor(dataset.embeddings).to(self.device))
        self.embeddings.weight.requires_grad = self.FINE_TUNE_EMBEDDINGS

        # Train the model
        for epoch in range(self.NUM_EPOCHS):
            # Get the initial state of the model
            state = self.get_init_state()
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0

            # Mini-batch training
            for X, y in dataloader:
                # Zero the gradient
                optimizer.zero_grad()
                # Create a tensor for the input and the target
                X = torch.LongTensor(X).to(self.device)
                y = torch.LongTensor(y).to(self.device)
                # Forward pass
                y_pred, state = self.forward(X, state)
                # Compute the loss
                loss = loss_fn(y_pred, y)

                # Detach the state so that it doesn't contribute to the gradient
                state = state.detach()

                # Backward pass
                loss.backward()
                # Update the parameters
                optimizer.step()
                # Contribute to the epoch loss and accuracy
                epoch_loss += loss.item()

                # Compute the accuracy
                epoch_accuracy += torch.sum(torch.argmax(y_pred, dim=1) == y).item() / y.shape[0]
                
                num_batches += 1

                # Free up memory
                del X, y, loss
                # Clear cache
                torch.cuda.empty_cache()

            # Average the epoch loss and accuracy
            epoch_loss /= num_batches
            epoch_accuracy /= num_batches
            print(f"Epoch {epoch + 1}/{self.NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            # Log the loss and accuracy to Weights and Biases
            wandb.log({"training_loss": epoch_loss, "training_accuracy": epoch_accuracy})
    
    # Inference function
    def predict(self, dataset, prompt, num_tokens):
        tokenizer = get_tokenizer("basic_english")
        # Convert the prompt to a list of tokens
        tokens = tokenizer(prompt)
        # Convert the tokens to indices into the vocabulary
        tokens = dataset.indexify([tokens], dataset.vocab)[0]

        for _ in range(num_tokens):
            with torch.no_grad():
                # Create a tensor for the input
                X = torch.LongTensor(tokens[-self.SEQUENCE_LENGTH:]).to(self.device)
                # Add a batch dimension
                X = X.unsqueeze(0)
                output, _ = self.forward(X, self.get_init_state())
            # Sample the next token
            tokens.append(output.argmax(dim=1).item())

        return dataset.unindexify(tokens, dataset.vocab)

