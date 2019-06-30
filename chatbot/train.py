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
import yaml
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--max_length', default=30, type=int,
                    help='maximum length of training sentences')
parser.add_argument('--config', default='./config/base.yml',
                    help='config file for Transformer')
parser.add_argument('--num_examples', default=-1, type=int,
                    help='number of training pairs')
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--checkpoint_dir', default='checkpoints',
                    help='directory to store checkpoints')
args = parser.parse_args()


if __name__ == '__main__':
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    dataset, info = create_dataset(
        args.max_length, config['BATCH_SIZE'], args.num_examples)

    encoder, decoder = create_transformer(
        info['vocab_size'], config['MODEL_SIZE'],
        info['max_length'], config['NUM_LAYERS'], config['H'])


    os.makedirs(args.checkpoint_dir, exist_ok=True)
    encoder_ckpt_path = os.path.join(
        args.checkpoint_dir, 'encoder_epoch_{}.h5')
    decoder_ckpt_path = os.path.join(
        args.checkpoint_dir, 'decoder_epoch_{}.h5')

    encoder_ckpts = glob.glob(os.path.join(args.checkpoint_dir, 'encoder*.h5'))
    decoder_ckpts = glob.glob(os.path.join(args.checkpoint_dir, 'decoder*.h5'))
    epoch_start = 0
    
    if len(encoder_ckpts) > 0 and len(decoder_ckpts) > 0:
        latest_encoder_ckpt = max(encoder_ckpts, key=os.path.getctime)
        encoder.load_weights(latest_encoder_ckpt)
        latest_decoder_ckpt = max(decoder_ckpts, key=os.path.getctime)
        decoder.load_weights(latest_decoder_ckpt)
        epoch_start = int(latest_encoder_ckpt[latest_encoder_ckpt.rfind('_')+1:-3])

    num_steps = info['data_size'] // config['BATCH_SIZE']
    print('num steps', num_steps)
    optimizer = create_optimizer(config['MODEL_SIZE'], trained_steps=epoch_start*num_steps)
    
    starttime = time.time()
    for e in range(epoch_start, args.num_epochs):
        avg_loss = 0.0
        for i, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
            loss = train_step(source_seq, target_seq_in, target_seq_out,
                              encoder, decoder, optimizer)
            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)

            if (i + 1) % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
                    e + 1, i + 1, avg_loss, time.time() - starttime))
                starttime = time.time()

        encoder.save_weights(encoder_ckpt_path.format(e + 1))
        decoder.save_weights(decoder_ckpt_path.format(e + 1))

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
