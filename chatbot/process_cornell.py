import os
import itertools
from collections import Counter
import re

rootpath = '../data/cornell'
movie_conversations = os.path.join(rootpath, 'movie_conversations.txt')
movie_lines = os.path.join(rootpath, 'movie_lines.txt')


def process_line(line):
    line = re.sub(r'([.,!?])', r' \1', line)
    line = re.sub(r'[^a-zA-Z.,!?\']+', r' ', line)
    line = line.strip()
    return line


def create_training_files():
    line_dict = {}
    words = []
    with open(movie_lines, 'r', encoding='iso-8859-1') as f:
        for line in f:
            line = line.split(' +++$+++ ')
            processed_line = process_line(line[-1])
            line_dict[line[0]] = processed_line
            words.append(processed_line.split())

    input_sequence = []
    target_sequence = []
    with open(movie_conversations, 'r') as f:
        for line in f:
            line = line.split(' +++$+++ ')
            line_ids = eval(line[-1])
            for i in range(0, len(line_ids) - 1):
                # if i + 1 == len(line_ids):
                #     break
                input_sequence.append(line_dict[line_ids[i]])
                target_sequence.append(line_dict[line_ids[i + 1]])

    with open('processed_input_data.txt', 'w', encoding='utf-8') as f:
        for sent in input_sequence:
            f.write(sent)
            f.write('\n')

    with open('processed_target_data.txt', 'w', encoding='utf-8') as f:
        for sent in target_sequence:
            f.write(sent)
            f.write('\n')

    words = list(itertools.chain(*words))
    word_counts = Counter(words)
    word_counts = [k for (k, v) in dict(filter(lambda x: x[1] > 10, tuple(word_counts.items()))).items()]
    print('Number of unique words:', len(word_counts))

    with open('vocab.txt', 'w', encoding='utf-8') as f:
        f.write('<unk>\n')
        f.write('<sos>\n')
        f.write('<eos>\n')
        for word in word_counts:
            f.write(word)
            f.write('\n')

if __name__ == '__main__':
    create_training_files()


