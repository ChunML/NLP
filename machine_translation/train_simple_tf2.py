import tensorflow as tf
import numpy as np
import unicodedata
import re

raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('What about his girlfriend?', "Qu'en est-il de sa copine ?"),
    ('What an inspiring speaker!', 'Quel brillant orateur !'),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ('What time did you wake up?', "À quelle heure t'es-tu réveillé ?"),
    ('When do you go on holiday?', "Quand pars-tu en vacances ?"),
    ('Who else knows the answer?', "Qui d'autre connaît la réponse ?"),
    ('He got up earlier than usual.', "Il s'est levé plus tôt que d'habitude."),
    ('He is the oldest of them all.', "C'est le plus vieux d'entre eux."),
    ('He left school two weeks ago.', "Il a obtenu son diplôme de fin d'année il y a 2 semaines."),
    ('He reached Kyoto on Saturday.', "Il est arrivé samedi à Kyoto."),
    ('He refused my friend request.', "Il a refusé ma demande pour devenir amis."),
    ('He refused to take the bribe.', "Il s'est refusé à prendre le pot-de-vin."),
    ('He studied the way birds fly.', "Il étudiait la manière dont les oiseaux volent."),
    ('He wondered why she did that.', "Il se demanda pourquoi elle avait fait cela."),
    ('Health is better than wealth.', "La santé est plus importante que la richesse."),
    ('Her face suddenly turned red.', "Son visage vira soudain au rouge."),
    ('His ideas conflict with mine.', "Ses idées rentrent en conflit avec les miennes."),
    ('I advised Tom not to do that.', "J'ai conseillé à Tom de ne pas faire cela."),
    ("I didn't know you were that good at French.", "J'ignorais que vous étiez aussi bonnes en français.")
)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')
        


def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s

raw_data_en, raw_data_fr = list(zip(*raw_data))
raw_data_en, raw_data_fr = list(raw_data_en), list(raw_data_fr)
raw_data_en = ['<start> ' + data + ' <end>' for data in raw_data_en]
raw_data_fr_in = ['<start> ' + data for data in raw_data_fr]
raw_data_fr_out = [data + ' <end>' for data in raw_data_fr]

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(1000, filters='')
en_tokenizer.fit_on_texts(raw_data_en)
data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,
                                                        padding='post')

fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(1000, filters='')
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)
data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in,
                                                           padding='post')

data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,
                                                            padding='post')

dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(20).batch(5)

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True)
    
    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)
        
        return output, state_h, state_c
    
    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))
    

en_vocab_size = len(en_tokenizer.word_index) + 1
EMBEDDING_SIZE = 32
LSTM_SIZE = 64

encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Decoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
        
    def call(self, sequence, state):
        embed = self.embedding(sequence)
        lstm_out, state_h, state_c = self.lstm(embed, state)
        logits = self.dense(lstm_out)
        
        return logits, state_h, state_c
    
    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))
    
fr_vocab_size = len(fr_tokenizer.word_index) + 1
decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)

def loss_func(targets, logits):
    logits = tf.squeeze(logits, 1)
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(logits, 0))
    loss = crossentropy(targets, logits)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    return tf.reduce_mean(loss)

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    loss = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_states = en_states

        for i in range(data_fr_out.shape[1]):
            de_outputs = decoder(tf.expand_dims(target_seq_in[:, i], 1), de_states)
        
            logits = de_outputs[0]
            de_states = de_outputs[1:]
        
            loss += loss_func(target_seq_out[:, i], logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return loss / data_fr_out.shape[1]

NUM_EPOCHS = 300
BATCH_SIZE = 5

for e in range(NUM_EPOCHS):
    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        en_initial_states = encoder.init_states(BATCH_SIZE)
        loss = train_step(source_seq, target_seq_in, target_seq_out, en_initial_states)
        
    print('Epoch {} Loss {:.4f}'.format(e + 1, loss.numpy()))

test_source_text = raw_data_en[np.random.choice(20)]
print(test_source_text)
test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
print(test_source_seq)

en_initial_states = encoder.init_states(1)
en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

de_input = tf.constant([[fr_tokenizer.word_index['<start>']]])
de_state_h, de_state_c = en_outputs[1:]
out_words = []

while True:
    de_output, de_state_h, de_state_c = decoder(de_input, (de_state_h, de_state_c))
    de_input = tf.argmax(de_output, -1)
    out_words.append(fr_tokenizer.index_word[de_input.numpy()[0][0]])
    
    if out_words[-1] == '<end>':
        break
        
print(' '.join(out_words))


