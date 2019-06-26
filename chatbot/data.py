from data_utils import maybe_download_and_read_file, normalize_string
import tensorflow as tf


URL = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
FILENAME = 'cornell_movie_dialogs_corpus.zip'


def create_dataset(max_length, batch_size, num_examples=-1):
    raw_data = maybe_download_and_read_file(URL, FILENAME)
    raw_data = list(filter(lambda x: len(
        x[0]) != 0 or len(x[0]) != 0, raw_data))
    raw_data = list(filter(lambda x: len(x[0].split()) <= max_length and len(
        x[1].split()) <= max_length, raw_data))

    """## Preprocessing"""
    if num_examples != -1:
        raw_data = raw_data[:num_examples]
    raw_input_lines, raw_target_lines = list(zip(*raw_data))
    raw_input_lines = [normalize_string(data) for data in raw_input_lines]
    raw_target_lines_in = ['<start> ' +
                           normalize_string(data) for data in raw_target_lines]
    raw_target_lines_out = [normalize_string(
        data) + ' <end>' for data in raw_target_lines]

    """## Tokenization"""

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(raw_input_lines)
    tokenizer.fit_on_texts(raw_target_lines_in)
    tokenizer.fit_on_texts(raw_target_lines_out)
    input_lines = tokenizer.texts_to_sequences(raw_input_lines)
    input_lines = tf.keras.preprocessing.sequence.pad_sequences(
        input_lines,
        padding='post')

    target_lines_in = tokenizer.texts_to_sequences(raw_target_lines_in)
    target_lines_in = tf.keras.preprocessing.sequence.pad_sequences(
        target_lines_in,
        padding='post')

    target_lines_out = tokenizer.texts_to_sequences(raw_target_lines_out)
    target_lines_out = tf.keras.preprocessing.sequence.pad_sequences(
        target_lines_out,
        padding='post')

    """## Create tf.data.Dataset object"""
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_lines, target_lines_in, target_lines_out))
    dataset = dataset.shuffle(len(input_lines)).batch(batch_size)

    """## Create the Positional Embedding"""

    max_length = max(len(input_lines[0]), len(target_lines_in[0]))
    vocab_size = len(tokenizer.word_index) + 1

    info = {
        'max_length': max_length,
        'vocab_size': vocab_size,
        'tokenizer': tokenizer
    }

    return dataset, info
