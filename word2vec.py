from collections import Counter
import collections
import tensorflow as tf
import numpy as np
import math
import random

#build a dataset with numbers representing words
stoplist = set({"(",")",".",",","?",":","--","-"})
vocabulary_size = 50000
words = open("training-data.1m").read().split()
words = filter(lambda a: a not in stoplist, words)
wordcounts = Counter(words).most_common(vocabulary_size - 1)
wordcounts += [("UNK", 0)]
print "Our vocab is " + str(len(wordcounts)) + " words long"
print "Top 5 words are : " + str(wordcounts[0:5])
vocab = dict([(a[0], i) for i, a in enumerate(wordcounts)])
rev_vocab = dict(zip(vocab.values(), vocab.keys()))

data = list()
for line in open("training-data.1m"):
    words = filter(lambda a: a not in stoplist, line)
    dataline = list()
    for word in line.split():
        dataline += [vocab[word] if word in vocab else vocabulary_size - 1]
    data += [dataline]

data_index = [0, 0]

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  batch_idx = 0

  while True:
    sentence_idx = data_index[0]
    word_idx = data_index[1]
    sentence = data[sentence_idx]

    left = max(0, word_idx-skip_window)
    right = min(len(sentence), word_idx+skip_window+1)
    target = word_idx

    for i in range(left, right):
      if i == target:
        continue
      batch[batch_idx] = sentence[word_idx]
      labels[batch_idx, 0] = sentence[i]
      batch_idx += 1
      if batch_idx == batch_size:
        return batch, labels

    if data_index[1] == len(sentence) - 1:
      data_index[1] = 0
      data_index[0] = (data_index[0] + 1) % len(data)
    else:
      data_index[1] += 1

#for _ in range(10000):
#  print ""
#  batch, labels = generate_batch(8, 2, 1)
#  for b, l in zip(batch, labels):
#    print rev_vocab[b] + "->" + rev_vocab[l[0]]

#SKIPGRAM MODEL NOTE: Portions of this part of the code were taken from another implementation
#of skipgrams using tensorflow.  We were going to write our own implementation but we determined
#via this example code that our approach and feature selection was inadequate, so we just
#include this code to show what we tried.  Our actual best results are from a model written
#entirely by us.
batch_size = 512
embedding_size = 512 # Dimension of the embedding vector.
skip_window = 4       # How many words to consider left and right.
num_skips = 8         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
valid_examples = [a + 50 for a in valid_examples]
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  valid_dataset2 = tf.constant([200], dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                     num_sampled, vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm

  fbi_embedding = tf.nn.embedding_lookup(normalized_embeddings, [vocab["FBI"]])
  fingerprint_embedding = tf.nn.embedding_lookup(normalized_embeddings, [vocab["fingerprint"]])
  stock_embedding = tf.nn.embedding_lookup(normalized_embeddings, [vocab["stock"]])
  string_embedding = tf.nn.embedding_lookup(normalized_embeddings, [vocab["string"]])
  bird_embedding = tf.nn.embedding_lookup(normalized_embeddings, [vocab["bird"]])
  crane_embedding = tf.nn.embedding_lookup(normalized_embeddings, [vocab["crane"]])
  computer_embedding = tf.nn.embedding_lookup(normalized_embeddings, [vocab["computer"]])
  software_embedding = tf.nn.embedding_lookup(normalized_embeddings, [vocab["software"]])

  similarity_fbi_fingerprint = tf.matmul(fbi_embedding, fingerprint_embedding, transpose_b=True)
  similarity_stock_string = tf.matmul(stock_embedding, string_embedding, transpose_b=True)
  similarity_bird_crane = tf.matmul(bird_embedding, crane_embedding, transpose_b=True)
  similarity_computer_software = tf.matmul(computer_embedding, software_embedding, transpose_b=True)

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  tf.initialize_all_variables().run()
  print("Initialized")

  average_loss = 0

  #for each step in the optimization, generate a new batch of data
  for step in xrange(num_steps):
    #get a batch of 128 skip-grams to try and optimize
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # update the values for the context vector and the embedding vector
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      print "FBI - fingerprint: " + str(similarity_fbi_fingerprint.eval())
      print "stock - string: " + str(similarity_stock_string.eval())
      print "bird - crane: " + str(similarity_bird_crane.eval())
      print "computer_software: " + str(similarity_computer_software.eval())

  # Generate output file
  out = open("output", "w+")
  final_embeddings = normalized_embeddings.eval()
  out.write(str(len(vocab)) + " " + str(embedding_size) + "\n")
  for i, e in enumerate(final_embeddings):
    out.write(rev_vocab[i] + " " + " ".join(e.astype('|S10'))+"\n")
