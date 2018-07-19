from tqdm import tqdm
import pickle
import time
import shelve
import os
os.sys.path.append('../../../')

import gensim
import chainer
from chainer import cuda
from chainer import serializers
import chainer.optimizers as O
import numpy as np
from lda2vec.logging import logger

data_dir = os.getenv('data_dir', '../data/')
fn_vocab = '{data_dir:s}/vocab.pkl'.format(data_dir=data_dir)
fn_corpus = '{data_dir:s}/corpus.pkl'.format(data_dir=data_dir)
fn_flatnd = '{data_dir:s}/flattened.npy'.format(data_dir=data_dir)
fn_docids = '{data_dir:s}/doc_ids.npy'.format(data_dir=data_dir)
fn_vectors = '{data_dir:s}/vectors.npy'.format(data_dir=data_dir)
fn_bow = '{data_dir:s}/bow.npy'.format(data_dir=data_dir)
fn_pruned = '{data_dir:s}/pruned.npy'.format(data_dir=data_dir)
fn_docweights = '{data_dir:s}/doc_weights_init.npy'.format(data_dir=data_dir)

vocab = pickle.load(open(fn_vocab, 'rb'))
corpus = pickle.load(open(fn_corpus, 'rb'))
flattened = np.load(fn_flatnd)
doc_ids = np.load(fn_docids)
vectors = np.load(fn_vectors)
bow = np.load(fn_bow)
pruned = np.load(fn_pruned)

n_topics = int(os.getenv('n_topics', 20))

# Number of unique words in the vocabulary
n_vocab = flattened.max() + 1
# A list of strings representations corresponding to word indices zero to `max_compact_index`
words = corpus.word_list(vocab)[:n_vocab]
dictionary={ compact_id:word_string for compact_id,word_string in zip(range(n_vocab), words) }
assert(dictionary[0] == '<SKIP>')
assert(dictionary[1] == 'out_of_vocabulary')
assert(dictionary[2] == '-PRON-')

# texts = [[decoder[j] for j in doc] for i, doc in encoded_docs]
# dictionary = corpora.Dictionary(texts)
# corpus = [dictionary.doc2bow(text) for text in texts]
logger.info('Start preparing corpus bow by document ...')
corpus = []
for row_idx in tqdm(range(bow.shape[0])):
    bow_row = bow[row_idx, :]
    corpus.append([ (compact_id, word_frequency) for compact_id, word_frequency in enumerate(bow_row) ])

logger.info('Start gensim LDA modeling ...')
lda = gensim.models.LdaModel(corpus, alpha=0.9, id2word=dictionary, num_topics=n_topics)
corpus_lda = lda[corpus]

doc_weights_init = np.zeros((len(corpus_lda), n_topics))

logger.info('Constructing document topic weights ...')
for i in tqdm(range(len(corpus_lda))):
    topics = corpus_lda[i]
    for j, prob in topics:
        doc_weights_init[i, j] = prob

# (Pdb) doc_weights_init.shape -> (11314, 20)
assert 0.9 < np.sum(doc_weights_init[0,:]) <= 1
assert 0.9 < np.sum(doc_weights_init[1,:]) <= 1
assert 0.9 < np.sum(doc_weights_init[2,:]) <= 1

np.save(fn_docweights, doc_weights_init)
