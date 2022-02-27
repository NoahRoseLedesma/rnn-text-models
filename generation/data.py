import torch
from torchtext.data.utils import get_tokenizer
import numpy as np
from tqdm import tqdm
import os
import pickle

GLOVE_PATH = "data/glove.840B.300d.txt"
GLOVE_VOCAB_SIZE = 2196017
GLOVE_EMBEDDING_SIZE = 300

class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, sequence_length):
        self.data_path = data_path
        self.sequence_length = sequence_length

        # Load the plaintext jokes into a monolithic corpus
        corpus = self.load_data()
        # Create a vocabulary of all of the words in the corpus
        self.word_freq = self.load_vocab(corpus)
        self.vocab = list(self.word_freq.keys())

        if os.path.exists("cache/embeddings.pkl"):
            print("Loading embeddings from cache...")
            # Load the embeddings from the pickle
            with open("cache/embeddings.pkl", 'rb') as f:
                self.embeddings = np.array(pickle.load(f))
        else:
            # Load the word embeddings
            full_glove_embeddings = self.load_glove()
            self.embeddings = self.make_embeddings(full_glove_embeddings)
            # Save the embeddings to a pickle
            with open("cache/embeddings.pkl", 'wb') as f:
                pickle.dump(self.embeddings, f)

        # Create a list of token indices for each joke
        self.indicies = self.indexify(corpus, self.vocab)

    def __iter__(self):
        self.cur_joke = 0
        self.joke_pos = 0

        return self

    def __next__(self):
        # Check if we've reached the end of the dataset
        if self.cur_joke == len(self.indicies):
            raise StopIteration()
        
        end_pos = self.joke_pos + self.sequence_length

        # Check if the current joke is over
        if end_pos >= len(self.indicies[self.cur_joke]):
            self.cur_joke += 1
            self.joke_pos = 0
            return next(self)

        X = self.indicies[self.cur_joke][self.joke_pos:end_pos]
        y = self.indicies[self.cur_joke][end_pos]
        
        # Update the joke position
        self.joke_pos += 1

        return torch.tensor(X), torch.tensor(y)

    # Load the GloVe embeddings
    def load_glove(self):
        embeddings = {}
        print("Loading GloVe embeddings...")
        with open(GLOVE_PATH, 'r') as f:
            for line in tqdm(f, total=GLOVE_VOCAB_SIZE):
                values = line.split(" ")
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                embeddings[word] = embedding
        # Add the unknown token
        embeddings['[UNK]'] = np.random.normal(0, 1, GLOVE_EMBEDDING_SIZE)
        return embeddings
    
    # Returns a list of embeddings for each word in the vocab
    def make_embeddings(self, full_glove_embeddings):
        embedding_weights = []
        for word in self.vocab:
            try:
                embedding_weights.append(full_glove_embeddings[word])
            except KeyError:
                embedding_weights.append(full_glove_embeddings['[UNK]'])

        return np.array(embedding_weights)
    
    def load_data(self):
        tokenizer = get_tokenizer("basic_english")
        corpus = []
        with open(self.data_path, 'r') as f:
            # Skip the first line
            f.readline()
            # Read the rest of the file
            for line in f:
                line = line.split(',')[1]
                # Remove leading and trailing quotes
                line = line[1:-2]
                # Tokenize the joke
                tokens = tokenizer(line)
                # Add the tokens to the list of jokes
                corpus.append(tokens)
        return corpus

    def load_vocab(self, corpus):
        # Find all of the unique words in the corpus
        vocab = {}
        for joke in corpus:
            for token in joke:
                if token not in vocab:
                    vocab[token] = 1
                else:
                    vocab[token] += 1
        # Ensure that the OOV token is in the vocab
        vocab['[UNK]'] = 1
        return vocab
    
    def indexify(self, corpus, vocab):
        indices = []
        for joke in corpus:
            joke_indices = []
            for token in joke:
                try:
                    joke_indices.append(vocab.index(token))
                except ValueError:
                    # Use a special token for words not in the vocab
                    joke_indices.append(vocab.index('[UNK]'))
            indices.append(joke_indices)
        return indices
    
    def unindexify(self, indices, vocab):
        return [vocab[index] for index in indices]