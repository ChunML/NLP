import tensorflow as tf
import numpy as np
import argparse
import os
import requests
from zipfile import ZipFile
import time
from model import create_transformer
from training_utils import train_step, create_optimizer
from data import create_dataset
from test import predict


parser = argparse.ArgumentParser()
parser.add_argument('--max_length', default=30, type=int,
                    help='maximum length of training sentences')
parser.add_argument('--model_size', default=128, type=int,
                    help='model size of the Transformer')
parser.add_argument('--h', default=4, help='number of attention heads')
parser.add_argument('--num_layers', default=3,
                    type=int, help='number of layers')
parser.add_argument('--num_examples', default=-1, type=int,
                    help='number of training pairs')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
args = parser.parse_args()


if __name__ == '__main__':
    dataset, info = create_dataset(
        args.max_length, args.batch_size, args.num_examples)

    encoder, decoder = create_transformer(
        info['vocab_size'], args.model_size,
        info['max_length'], args.num_layers, args.h)

    optimizer = create_optimizer(args.model_size)

    starttime = time.time()
    for e in range(args.num_epochs):
        avg_loss = 0.0
        for i, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
            loss = train_step(source_seq, target_seq_in, target_seq_out,
                              encoder, decoder, optimizer)
            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)

            if (i + 1) % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
                    e + 1, i + 1, avg_loss, time.time() - starttime))
                starttime = time.time()

        encoder.save_weights('./checkpoints/encoder_epoch_{}'.format(e + 1))
        decoder.save_weights('./checkpoints/decoder_epoch_{}'.format(e + 1))

        try:
            predict(encoder, decoder, info['tokenizer'],
                    ['Hey , how are you today ?',
                     'So you said you don t want to join us tonight ?',
                     'I love you .',
                     'It s not your fault , Steve .',
                     'Where are you now ?'],
                    info['max_length'] * 2)
        except Exception as e:
            print(e)
            continue
