import numpy as np
import tensorflow as tf
import argparse
import yaml
from data import create_dataset
from model import create_transformer


def predict(encoder, decoder, tokenizer, raw_input_lines, max_length):
    """ Predict the output sentence for a given input sentence

    Args:
        test_source_text: input sentence (raw string)

    Returns:
        The encoder's attention vectors
        The decoder's bottom attention vectors
        The decoder's middle attention vectors
        The input string array (input sentence split by ' ')
        The output string array
    """
    test_source_text = np.random.choice(raw_input_lines)
    # print(test_source_text)
    test_source_seq = tokenizer.texts_to_sequences([test_source_text])
    # print(test_source_seq)

    en_output, en_alignments = encoder(
        tf.constant(test_source_seq), training=False)

    de_input = tf.constant(
        [[tokenizer.word_index['<start>']]], dtype=tf.int64)

    out_words = []

    while True:
        de_output, de_bot_alignments, de_mid_alignments = decoder(
            de_input, en_output, training=False)
        new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
        out_words.append(tokenizer.index_word[new_word.numpy()[0][0]])

        # Transformer doesn't have sequential mechanism (i.e. states)
        # so we have to add the last predicted word to create a new input sequence
        de_input = tf.concat((de_input, new_word), axis=-1)

        # TODO: get a nicer constraint for the sequence length!
        if out_words[-1] == '<end>' or len(out_words) >= max_length:
            break

    print('Bot: ' + ' '.join(out_words))
    return en_alignments, de_bot_alignments, de_mid_alignments, test_source_text.split(' '), out_words


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', default=30, type=int,
                        help='maximum length of training sentences')
    parser.add_argument('--config', default='./config/base.yml',
                        help='config file for Transformer')
    parser.add_argument('--encoder_weights_path', required=True,
                        help='path to encoder weights')
    parser.add_argument('--decoder_weights_path', required=True,
                        help='path to decoder weights')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    # TODO: Is there any smarter way to get the vocab size?
    _, info = create_dataset(
        args.max_length, 1, 100)

    encoder, decoder = create_transformer(
        info['vocab_size'], config['MODEL_SIZE'],
        info['max_length'], config['NUM_LAYERS'], config['H'])

    encoder.load_weights(args.encoder_weights_path)
    decoder.load_weights(args.decoder_weights_path)

    input_sentence = [input('Chun: ')]

    while input_sentence[0] != 'quit':
        predict(encoder, decoder, info['tokenizer'],
                input_sentence, info['max_length'])

        input_sentence = [input('Chun: ')]
