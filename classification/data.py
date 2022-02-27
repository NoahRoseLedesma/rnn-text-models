# Provides an interface to loading the training and testing data
import os
import numpy as np
import itertools
import re

POS_TRAIN_DATA_PREFIX = 'data/embedded/train_pos_'
NEG_TRAIN_DATA_PREFIX = 'data/embedded/train_neg_'
POS_TEST_DATA_PREFIX = 'data/embedded/test_pos_'
NEG_TEST_DATA_PREFIX = 'data/embedded/test_neg_'

# The number of tokens in each review
REVIEW_SIZE = 200

# Provide an interface to get batches of pre-embedded data
class BatchIterator():
    def __init__(self, batch_size, pos_path, neg_path, embedding_size):
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.instance_size = embedding_size * REVIEW_SIZE * 4
        self.pos_file = open(pos_path, 'rb')
        self.neg_file = open(neg_path, 'rb')
        self.embedding_size = embedding_size
        self.current_batch = 1
        # Find the number of reviews in the file
        num_pos = os.path.getsize(pos_path) // self.instance_size
        num_neg = os.path.getsize(neg_path) // self.instance_size
        assert num_pos == num_neg
        self.num_reviews = num_pos
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.num_reviews // self.batch_size

    def __next__(self):
        if self.current_batch == len(self):
            raise StopIteration
        
        # Get a half batch of positive reviews
        pos_batch = self.pos_file.read(self.instance_size * self.batch_size // 2)
        pos_batch = np.frombuffer(pos_batch, dtype=np.float32)
        pos_batch = pos_batch.reshape(self.batch_size // 2, REVIEW_SIZE, self.embedding_size)
        # Get a half batch of negative reviews
        neg_batch = self.neg_file.read(self.instance_size * self.batch_size // 2)
        neg_batch = np.frombuffer(neg_batch, dtype=np.float32)
        neg_batch = neg_batch.reshape(self.batch_size // 2, REVIEW_SIZE, self.embedding_size)
        # Concatenate the batches
        batch = np.concatenate((pos_batch, neg_batch), axis=0)
        # Create the labels
        labels = np.concatenate((np.ones(self.batch_size // 2), np.zeros(self.batch_size // 2)))
        # Shuffle the data
        indices = np.arange(self.batch_size)
        np.random.shuffle(indices)
        batch = batch[indices]
        labels = labels[indices]

        self.current_batch += 1
        return batch, labels
    
    def reset(self):
        self.pos_file.seek(0)
        self.neg_file.seek(0)
        self.current_batch = 1

    def __del__(self):
        self.pos_file.close()
        self.neg_file.close()

def load_data(batch_size, embedding_size):
    # Determine the file paths for the training and testing data
    train_pos_path = POS_TRAIN_DATA_PREFIX + str(embedding_size) + 'd.bin'
    train_neg_path = NEG_TRAIN_DATA_PREFIX + str(embedding_size) + 'd.bin'
    test_pos_path = POS_TEST_DATA_PREFIX + str(embedding_size) + 'd.bin'
    test_neg_path = NEG_TEST_DATA_PREFIX + str(embedding_size) + 'd.bin'

    return {
        'train': BatchIterator(batch_size, train_pos_path, train_neg_path, embedding_size),
        'test': BatchIterator(batch_size, test_pos_path, test_neg_path, embedding_size)
    }
