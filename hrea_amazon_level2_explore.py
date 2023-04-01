# explore bank obtained in level 2 model

import torch
import numpy as np
import pickle
from hrea_amazon_txtprocess import WordEmbeds

# hyper-parameters
topn = 15

filename = 'encoder_parameters.pl'
infile = open(filename, 'rb')
parameters_dic = pickle.load(infile)
infile.close()
embed_dim = parameters_dic['embedding_dim']
enc_dim = parameters_dic['encoder_enc_dim']

# get bank and word embeddings
embed_file = 'amazon-embed.txt'

with open('bank1.out', 'rb') as f:
    bank1 = pickle.load(f)

with open('bank2.out', 'rb') as f:
    bank2 = pickle.load(f)

with open('vocab.pl', 'rb') as f:
    vocab = pickle.load(f)

nouns = vocab.nouns_id
idx2token = vocab.get_idx_to_token()
vocab_size = len(idx2token)
word_embeds = WordEmbeds(vocab.get_token_to_idx())
E = word_embeds.build_aspect_base(embed_file, nouns, embed_dim)
M = np.matmul(bank1, np.transpose(E))

idx_word = {}                       # index in nouns->word
for i in range(len(nouns)):
    idx_word[i] = idx2token[nouns[i]]

# find top n similar words for a cluster in bank1
print('\n------------- aspects clusters in bank1 -------------')
def find(vec, idx_word, topn):
    idx = np.argsort(vec)[::-1][:topn]
    return ([idx_word[id] for id in idx])

for i in np.arange(M.shape[0]):
    words = find(M[i], idx_word, topn)
    print(f'{i}: {words}')

# select topn words for bank2
print('\n------------ aspects clusters in bank2 ------------')
M2 = np.matmul(bank2, np.transpose(E))

def find2(vec, topn):
    idx = np.argsort(vec)[::-1][:topn]
    # return ([f'{id}:{vec[id]:0.2f}' for id in idx])
    return ([idx_word[id] for id in idx])

for i in np.arange(M2.shape[0]):
    words = find(M2[i], idx_word, topn)
    print(f'{i}: {words}')

print('\n------------- bank cluster in bank1 -----------')
# find top n features from bank 1
M3 = np.matmul(bank2, np.transpose(bank1))

def find3(vec, topn):
    idx = np.argsort(vec)[::-1][:topn]
    return ([f'{id}:{vec[id]:0.2f}' for id in idx])

for i in np.arange(M3.shape[0]):
    idx = find3(M3[i], 10)
    print(idx)