import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

SHUFFLE_SIZE = 1000
BATCH_SIZE = 64
EMBEDDING_SIZE = 64
LSTM_SIZE = 64
HIDDEN_SIZE = 64

imdb, info = tfds.load('imdb_reviews/subwords32k', with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

tokenizer = info.features['text'].encoder

train_data = train_data.shuffle(SHUFFLE_SIZE).padded_batch(BATCH_SIZE, train_data.output_shapes)

test_data = test_data.padded_batch(BATCH_SIZE, test_data.output_shapes)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, EMBEDDING_SIZE),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_SIZE)),
    tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu'),
    tf.keras.layers.Dense(1) #, activation='sigmoid')
])

loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
for e in range(10):
    accuracy = []
    for batch, (text, label) in enumerate(train_data.take(-1)):
        with tf.GradientTape() as tape:
            logits = model(text)
            label = tf.expand_dims(label, 1)
            loss = loss_func(label, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        grads_and_vars = zip(gradients, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars)
        predictions = tf.cast(tf.math.greater(tf.sigmoid(logits), 0.5), tf.int64)
        accuracy.extend(tf.cast(tf.equal(predictions, label), tf.int64).numpy())

        if batch % 100 == 0:
            print('\nEpoch: {} - Batch: {}'.format(e, batch))
            print('Loss: {:.4f}'.format(loss.numpy()))
            print('Accuracy: {}'.format(np.mean(accuracy)))
      
            for _, (text, label) in enumerate(test_data.take(1)):
                logits = model(text)
        
                random_id = np.random.choice(label.shape[0], 5)
                for ix in random_id:
                    print('\n')
                    print(tokenizer.decode(text.numpy()[ix]))
                    print('Label', u'\u2713' if label.numpy()[ix] == 1 else u'\u2715')
                    print('Pred', u'\u2713' if logits.numpy()[ix][0] >= 0.5 else u'\u2715')
          
          
test_text = "I caught this movie the other night on one of the movie channels and I haven't laughed that hard in a long time."
test_text = tokenizer.encode(test_text)

pred = model(tf.convert_to_tensor([test_text]))
print(u'\u2713' if pred.numpy()[0, 0] >= 0.5 else u'\u2715')
