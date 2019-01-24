# A Recap On Word Embeddings

## Overview
To Be Updated

## Installation
To Be Updated

## Implementation Details
### Input

In terms of word embedding, one (input, target) pair must contain a particular word and its surrounding words (in case of Skip-Gram, if CBOW is used, the order is reversed).

Ex: I have a good feeling tonight

- Skip-gram (with window size = 2, current word = *good*):

(input, target): (good, [have, a, feeling, tonight])

- CBOW (with window size = 2, current word = *good*):

(input, target): (have, good), (a, good), (feeling, good) (tonight, good)

### Network

The network to train word embedding is pretty simple (at least Skip-Gram and CBOW), we only need one Embedding Layer and one Dense Layer.

- Embedding Layer

Technically, Embedding layer is just a normal Dense layer, which takes the input vector of size `vocab_size` to a vector of size `embedding_size`. 

For example, let's assume that our data contains 3000 unique words. Each word will be a one-hot vector of size (1, 3000). We then pass it to an embedding layer with 300 hidden units. The result is a vector of size (1, 300), smaller size and no more sparse! Simple enough?

The code for embedding is like below, One notice though, we don't have to convert words to one-hot vectors by hand. But don't forget to convert words to integers. String is not good for machine ;)

```python
inputs = tf.placeholder(tf.int32, [batch_size])
embedding = tf.Variable(tf.random_uniform(
    [vocab_size, embedding_size], -1, 1))
embed = tf.nn.embedding_lookup(embedding, inputs)
```

- Dense Layer

We need a dense layer in order to compute logits, which is necessary for computing loss. Why?

Remember both input words and target words are of size `vocab_size`? The output of the embedding layer is a vector size `embedding_size`, so obviously, we need some way to project it back to `vocab_size` size. Dense layer comes in handy, right?

That's the theory. We won't explicitly implement a dense layer this time. To see why, let's talk about the loss.

### Loss & Training

In NLP problems, we usually have to deal with vectors of very large size (because of the `vocab_size`), which leads to heavy computational cost.

To make it worse, in word embedding training, there is only one ground-truth word among `vocab_size` words. You can do the math yourselves, but taking everyone into gradient computing is not a good idea (high cost, low efficience).

Solution for this? When computing gradients, we randomly use some of the negative words (together with the target word). This method has been proved to increase the training efficience, yet maintaining good results in practice. Read more [here](https://arxiv.org/abs/1412.2007){:target="_blank"}.

So here is how we implement in Tensorflow. Pay attention to the shape of the **weights**, it's `[vocab_size, embedding_size]`, not the other way around.

The method to use is `tf.nn.sampled_softmax_loss`, in which `num_sampled` specifies how many negative units we want to use to compute the loss. Setting it too small may cause the learning to fail.

```python
weights = tf.Variable(tf.truncated_normal(
    [vocab_size, embedding_size], stddev=0.1))
biases = tf.Variable(tf.zeros(vocab_size))
loss = tf.nn.sampled_softmax_loss(weights=weights,
                                  biases=biases,
                                  labels=labels,
                                  inputs=embed,
                                  num_sampled=n_sampled,
                                  num_classes=vocab_size)
cost = tf.reduce_mean(loss)
```

Training is pretty simple, we will use `AdamOptimizer` and think no more!

```python
optimizer = tf.train.AdamOptimizer().minimize(cost)
```

### Inference

## Reference
- Udacity's Embeddings: [link](https://github.com/udacity/deep-learning/tree/master/embeddings)
- Skip-gram Paper: [link](https://arxiv.org/abs/1310.4546)
- CBOW Paper: [link](https://arxiv.org/abs/1301.3781)
