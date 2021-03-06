# coding: utf-8

# Deep Learning
# =============
#
# Assignment 6
# ------------
#
# After training a skip-gram model in `5_word2vec.ipynb`, the goal of this notebook is to train a LSTM character model over [Text8](http://mattmahoney.net/dc/textdata) data.

# In[58]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
import collections
import math
import re
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
from layers import LSTM, FeedForward, ConnectLayers, SparseLSTM, TridirectionalHighway, HighwayLayer, GRU

# In[59]:

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')

    return filename


filename = maybe_download('text8.zip', 31344016)


# In[60]:

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        for name in f.namelist():
            return tf.compat.as_str(f.read(name))


text = read_data(filename)
print('Data size %d' % len(text))

# Create a small validation set.

# In[61]:

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

# Utility functions to map characters to vocabulary IDs and back.

# In[62]:

vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])


def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
    return 0


def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '


print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0))

# Function to generate a training batch for the LSTM model.

# In[63]:

batch_size = 32
num_unrollings = 20


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [ offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))

# In[64]:

def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10

    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1


def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p


def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b / np.sum(b, 1)[:, None]


# Simple LSTM Model.

# In[65]:

num_nodes2 = 150

def layerBlock(num_nodes, batch_size):
    return HighwayLayer(TridirectionalHighway(num_nodes, batch_size, 3, connection_mode='head'))

graph = tf.Graph()
with graph.as_default():
    layer1 = FeedForward(vocabulary_size, num_nodes2, batch_size)
    layer2 = ConnectLayers((layerBlock(num_nodes2, batch_size),
                            layerBlock(num_nodes2, batch_size),
                            layerBlock(num_nodes2, batch_size)))
    # layer2 = ConnectLayers((HighwayLayer(GRU(num_nodes2, num_nodes2, batch_size)),
    #                         HighwayLayer(GRU(num_nodes2, num_nodes2, batch_size)),
    #                         HighwayLayer(GRU(num_nodes2, num_nodes2, batch_size)),
    #                         HighwayLayer(GRU(num_nodes2, num_nodes2, batch_size)),
    #                         HighwayLayer(GRU(num_nodes2, num_nodes2, batch_size)),
    #                         LSTM(num_nodes2, num_nodes2, batch_size)))
    layer3 = FeedForward(num_nodes2, vocabulary_size, batch_size)
    model = ConnectLayers((layer1, layer2, layer3))
    # Input data.
    train_data = list()
    for _ in range(num_unrollings + 1):
        train_data.append(
            tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
    train_inputs = train_data[:num_unrollings]
    train_labels = train_data[1:]  # labels are inputs shifted by one time step.

    # Unrolled LSTM loop.
    outputs = list()
    for i in train_inputs:
        output = model.feed_input(i)
        outputs.append(output)

    # State saving across unrollings.
    with tf.control_dependencies([model.save_state()]):
        # Classifier.
        logits = tf.concat(0, outputs)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits, tf.concat(0, train_labels)))

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.constant(2.)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step)

    # global_step = tf.Variable(0)
    # learning_rate = tf.train.exponential_decay(
    #     10.0, global_step, 5000, 0.1, staircase=True)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # gradients, v = zip(*optimizer.compute_gradients(loss))
    # gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    # optimizer = optimizer.apply_gradients(
    #     zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
    sample_output = model.eval_model.feed_input(sample_input)
    reset_sample_state = model.eval_model.reset_saved_state()

    with tf.control_dependencies([model.eval_model.save_state()]):
        sample_prediction = tf.nn.softmax(sample_output)

# In[66]:

num_steps = 20000
summary_frequency = 100

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        batches = train_batches.next()
        feed_dict = dict()
        for i in range(num_unrollings + 1):
            feed_dict[train_data[i]] = batches[i]
        _, l, predictions, lr = session.run(
            [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
            # The mean loss is an estimate of the loss over the last few batches.
            print(
                'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            labels = np.concatenate(list(batches)[1:])
            print('Minibatch perplexity: %.2f' % float(
                np.exp(logprob(predictions, labels))))
            if step % (summary_frequency * 10) == 0:
                # Generate some samples.
                print('=' * 80)
                for _ in range(5):
                    feed = sample(random_distribution())
                    sentence = characters(feed)[0]
                    reset_sample_state.run()
                    for _ in range(79):
                        prediction = sample_prediction.eval({sample_input: feed})
                        feed = sample(prediction)
                        sentence += characters(feed)[0]
                    print(sentence)
                print('=' * 80)
            # Measure validation set perplexity.
            reset_sample_state.run()
            valid_logprob = 1e-4
            for _ in range(valid_size):
                b = valid_batches.next()
                predictions = sample_prediction.eval({sample_input: b[0]})
                valid_logprob = valid_logprob + logprob(predictions, b[1])
            print('Validation set perplexity: %.2f' % float(np.exp(
                valid_logprob / valid_size)))