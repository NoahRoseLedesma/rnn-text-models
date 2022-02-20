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

# Load the glove word embeddings
#def load_glove(path):
#    glove = {}
#    with open(path, 'r') as f:
#        for line in f:
#            splitLine = line.split()
#            word = splitLine[0]
#            embedding = np.array([float(val) for val in splitLine[1:]])
#            glove[word] = embedding
#    return glove

# Load training and testing data from the raw text files
#def load_data(embedding_size):
#    glove_path = f'./data/glove_embeddings/glove.6B.{embedding_size}d.txt'
#    # Regular expression to match html tags
#    html_regex = re.compile('<.*?>')
#    # Regular expression to match non-alphabetic (excluding dashes and spaces)
#    non_alphabetic_regex = re.compile('[^a-zA-Z\- ]')
#
#    # Load the glove embeddings
#    glove = load_glove(glove_path)
#
#    # Create a numpy array of size REVIEW_SIZE for the given review
#    def tokenize(review):
#        # Remove html tags
#        review = html_regex.sub('', review)
#        # Remove non-alphanumeric characters
#        review = re.sub(non_alphabetic_regex, '', review)
#        # Replace all dashes with spaces
#        review = review.replace('-', ' ')
#        # Convert to lowercase
#        review = review.lower()
#        # Splt the review into words
#        review = review.split()
#        # Get the word embeddings for the review
#        review = [glove[word] for word in review if word in glove]
#        # Truncate the review to REVIEW_SIZE
#        review = review[:REVIEW_SIZE]
#        # Pad the review to REVIEW_SIZE with the embedding for the padding token
#        review = review + [[0] * embedding_size] * (REVIEW_SIZE - len(review))
#        # Return the review as a numpy array
#        return np.array(review)
#
#    # Create an array of all the reviews in the provided directory
#    def get_reviews(data_path):
#        reviews = []
#        for f in os.listdir(data_path):
#            with open(data_path + f, 'r') as review_file:
#                # Get the review text
#                review = review_file.read()
#                # Tokenize the review and get the embeddings
#                reviews.append(tokenize(review))
#        return np.array(reviews)
#
#    # Get the reviews
#    train_pos = get_reviews(POS_TRAIN_DATA_PATH)
#    train_neg = get_reviews(NEG_TRAIN_DATA_PATH)
#    test_pos = get_reviews(POS_TEST_DATA_PATH)
#    test_neg = get_reviews(NEG_TEST_DATA_PATH)
#
#    # Convert the data to numpy arrays
#    train_data = np.concatenate((train_pos, train_neg))
#    train_labels = np.array([1] * len(train_pos) + [0] * len(train_neg))
#    test_data = np.concatenate((test_pos, test_neg))
#    test_labels = np.array([1] * len(test_pos) + [0] * len(test_neg))
#
#    # Shuffle the training data
#    indicies = np.arange(len(train_data))
#    np.random.shuffle(indicies)
#    train_data = train_data[indicies]
#    train_labels = train_labels[indicies]
#
#    return {
#        'train': {
#            'X': train_data,
#            'y': train_labels
#        },
#        'test': {
#            'X': test_data,
#            'y': test_labels
#        }
#    }
#