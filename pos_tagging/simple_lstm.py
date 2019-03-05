import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1234)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def prepare_char_sequence(word, to_ix):
    idxs = [to_ix[c] for c in word]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ('the dog ate the apple'.split(), ['DET', 'NN', 'V', 'DET', 'NN']),
    ('everybody read that book'.split(), ['NN', 'V', 'DET', 'NN'])]

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {'DET': 0, 'NN': 1, 'V': 2}
char_to_ix = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5,
              'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11,
              'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17,
              's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}

CHAR_EMBEDDING_DIM = 10
EMBEDDING_DIM = 6

CHAR_HIDDEN_DIM = 5
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, char_embedding_dim,
                 hidden_dim, char_hidden_dim,
                 vocab_size, char_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_hidden_dim = char_hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.char_embeddings = nn.Embedding(char_size, char_embedding_dim)

        self.lstm = nn.LSTM(embedding_dim + char_hidden_dim, hidden_dim)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        self.char_hidden = self.init_char_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def init_char_hidden(self):
        return (torch.zeros(1, 1, self.char_hidden_dim),
                torch.zeros(1, 1, self.char_hidden_dim))

    def forward(self, sentence):
        all_char_hidden = []
        for word in sentence:
            char_hidden = self.init_char_hidden()
            word = prepare_char_sequence(word, char_to_ix)
            char_embeds = self.char_embeddings(word)
            _, char_hidden = self.char_lstm(char_embeds.view(len(word), 1, -1), char_hidden)
            all_char_hidden.append(char_hidden[0])

        all_char_hidden = torch.cat(all_char_hidden).view(len(word), -1)

        sentence = prepare_sequence(sentence, word_to_ix)
        embeds = self.word_embeddings(sentence)
        embeds = torch.cat((embeds, all_char_hidden), dim=1)

        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, CHAR_EMBEDDING_DIM,
                   HIDDEN_DIM, CHAR_HIDDEN_DIM,
                   len(word_to_ix), len(char_to_ix),
                   len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(training_data[0][0])
    print(tag_scores)

for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()
        model.hidden = model.init_hidden()

        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(sentence)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    tag_scores = model(training_data[0][0])
    print(tag_scores)

