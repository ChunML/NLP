import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

TRAIN_FILE = '../data/quora/train.csv'
TEST_RATIO = 0.2
BUFFER_SIZE = 10000
BATCH_SIZE = 128
EMBEDDING_SIZE = 256
LSTM_SIZE = 512
HIDDEN_SIZE = 64

train_df, val_df = train_test_split(pd.read_csv(TRAIN_FILE), test_size=TEST_RATIO, shuffle=True)

train_questions, train_targets = train_df.question_text, train_df.target
train_questions, train_targets = list(train_questions), list(train_targets)

val_questions, val_targets = val_df.question_text, val_df.target
val_questions, val_targets = list(val_questions), list(val_targets)

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_questions)
tokenizer.fit_on_texts(val_questions)

train_sequences = tokenizer.texts_to_sequences(train_questions)
val_sequences = tokenizer.texts_to_sequences(val_questions)

train_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, padding='post')
val_sequences = tf.keras.preprocessing.sequence.pad_sequences(val_sequences, padding='post')

train_targets = np.array(train_targets)[:, None].astype(np.int64)
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_targets))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

val_targets = np.array(val_targets)[:, None].astype(np.int64)
val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_targets))
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(tokenizer.word_index) + 1

class RNNModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size, hidden_size):
        super(RNNModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size))
        self.dense = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.out = tf.keras.layers.Dense(1)

    def call(self, sequence):
        embed = self.embedding(sequence)
        lstm_out = self.lstm(embed)
        logits = self.out(self.dense(lstm_out))
        
        return logits
    
model = RNNModel(vocab_size, EMBEDDING_SIZE, LSTM_SIZE, HIDDEN_SIZE)

loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss = loss_func(targets, logits)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return logits, loss
    
def compute_f1(preds, labels):
    preds = tf.cast(preds, tf.bool)
    labels = tf.cast(labels, tf.bool)
    true_positive = tf.reduce_sum(tf.cast(tf.math.logical_and(preds, labels), tf.int64))
    false_positive = tf.reduce_sum(tf.cast(tf.math.logical_and(preds, tf.math.logical_not(labels)), tf.int64))
    false_negative = tf.reduce_sum(tf.cast(tf.math.logical_and(labels, tf.math.logical_not(preds)), tf.int64))
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * recall * precision / (recall + precision)
    
    return f1_score, precision, recall

def compute_accuracy(preds, labels):
    equals = tf.reduce_sum(tf.cast(tf.math.equal(preds, labels), tf.int64))
    accuracy = equals / preds.shape[0]
    
    return accuracy

print(compute_f1(tf.constant([0, 0, 1, 0, 1, 0]), tf.constant([1, 0, 1, 1, 0, 0])))
print(compute_accuracy(tf.constant([0, 0, 1, 0, 1, 0]), tf.constant([1, 0, 1, 1, 0, 0])))

NUM_EPOCHS = 1
steps_per_epoch = len(train_sequences) // BATCH_SIZE

for e in range(NUM_EPOCHS):
    for batch, (inputs, targets) in enumerate(train_dataset.take(steps_per_epoch)):
        logits, loss = train_step(inputs, targets)
        
        predictions = tf.cast(tf.math.greater_equal(tf.sigmoid(logits), 0.5), tf.int64)
        
        f1_score, precision, recall = compute_f1(predictions, targets)
        accuracy = compute_accuracy(predictions, targets)
        
        if batch % 100 == 0:
            print('Predictions', predictions.numpy()[:30, 0])
            print('Target     ', targets.numpy()[:30, 0])
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f} F1 {:.4f} Precision {:.4f} Recall {:.4f}\n'.format(
                e + 1, batch, loss.numpy(), accuracy.numpy(), f1_score.numpy(), precision.numpy(), recall.numpy()))
        
        if batch % 500 == 0:
            model.save_weights('epoch_{}_batch_{}.h5'.format(e + 1, batch)
            val_f1_score = 0.0
            val_accuracy = 0.0
            for batch, (inputs, targets) in enumerate(val_dataset.take(2)):
                val_logits = model(inputs)
                predictions = tf.cast(tf.math.greater_equal(tf.sigmoid(val_logits), 0.5), tf.int64)

                f1_score, _, _ = compute_f1(predictions, targets)
                accuracy = compute_accuracy(predictions, targets)
                
                val_f1_score = (val_f1_score + f1_score) / (batch + 1)
                val_accuracy = (val_accuracy + accuracy) / (batch + 1)

            print('=========================Validation============================')
            print('Predictions', predictions.numpy()[:30, 0])
            print('Target     ', targets.numpy()[:30, 0])
            print('Epoch {} Loss {:.4f} Accuracy {:.4f} F1 {:.4f}\n'.format(
                e + 1, loss.numpy(), val_accuracy.numpy(), val_f1_score.numpy()))
