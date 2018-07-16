# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import os
import os.path
import pickle
import time
import shelve

import chainer
from chainer import cuda
from chainer import serializers
import chainer.optimizers as O
import numpy as np

from lda2vec import utils
from lda2vec import prepare_topics, print_top_words_per_topic, topic_coherence
from lda2vec_model import LDA2Vec
from lda2vec.logging import logger
from lda2vec.utils import consine_distance

gpu_id = int(os.getenv('CUDA_GPU', -1))
if gpu_id >= 0:
    cuda.get_device(gpu_id).use()
    print("Using GPU " + str(gpu_id))
else:
    print("using CPU as environment variable CUDA_GPU is not defined")

data_dir = os.getenv('data_dir', '../data/')
fn_vocab = '{data_dir:s}/vocab.pkl'.format(data_dir=data_dir)
fn_corpus = '{data_dir:s}/corpus.pkl'.format(data_dir=data_dir)
fn_flatnd = '{data_dir:s}/flattened.npy'.format(data_dir=data_dir)
fn_docids = '{data_dir:s}/doc_ids.npy'.format(data_dir=data_dir)
fn_vectors = '{data_dir:s}/vectors.npy'.format(data_dir=data_dir)
vocab = pickle.load(open(fn_vocab, 'rb'))
corpus = pickle.load(open(fn_corpus, 'rb'))
flattened = np.load(fn_flatnd)
doc_ids = np.load(fn_docids)
vectors = np.load(fn_vectors)

# Model Parameters
# Number of documents
n_docs = doc_ids.max() + 1
# Number of unique words in the vocabulary
n_vocab = flattened.max() + 1
# 'Strength' of the dircihlet prior; 200.0 seems to work well
clambda = 200.0

# Number of topics to fit
n_topics = int(os.getenv('n_topics', 20))
batchsize = 4096

# Power for neg sampling
power = float(os.getenv('power', 0.75))

# Intialize with pretrained word vectors
pretrained = bool(int(os.getenv('pretrained', True)))

# Sampling temperature
temperature = float(os.getenv('temperature', 1.0))

# Number of dimensions in a single word vector
n_units = int(os.getenv('n_units', 300))

# Get the string representation for every compact key
# Inputs:
# vocab: dict from word long hash# (from Spacy) -> word string
# Outputs:
# words: dict from word compact index -> word string
words = corpus.word_list(vocab)[:n_vocab]

# How many tokens are in each document
doc_idx, lengths = np.unique(doc_ids, return_counts=True)
doc_lengths = np.zeros(doc_ids.max() + 1, dtype='int32')
doc_lengths[doc_idx] = lengths

# Count all token frequencies
tok_idx, freq = np.unique(flattened, return_counts=True)
term_frequency = np.zeros(n_vocab, dtype='int32')
term_frequency[tok_idx] = freq

for key in sorted(locals().keys()):
    val = locals()[key]
    if len(str(val)) < 100 and '<' not in str(val):
        print(key, val)

model = LDA2Vec(n_documents=n_docs, n_document_topics=n_topics,
                n_units=n_units, n_vocab=n_vocab, counts=term_frequency,
                n_samples=15, power=power, temperature=temperature, vocab=words)
if os.path.exists('lda2vec.hdf5'):
    print("Reloading from saved")
    serializers.load_hdf5("lda2vec.hdf5", model)

if pretrained:
    logger.info('Use pre-trained Google word2vec')
    model.sampler.W.data[:, :] = vectors[:n_vocab, :]

if gpu_id >= 0:
    model.to_gpu()
else:
    model.to_cpu()

optimizer = O.Adam()
# optimizer = O.SGD()
optimizer.setup(model)
clip = chainer.optimizer.GradientClipping(5.0)
optimizer.add_hook(clip)

j = 0
epoch = 0
fraction = batchsize * 1.0 / flattened.shape[0]
progress = shelve.open('progress.shelve')

for epoch in range(200):
    # After the first execution of the statement below, data.keys() =>
    # dict_keys(['vocab', 'doc_lengths', 'doc_topic_dists', 'topic_term_dists', 'term_frequency']
    #
    # Also the data['vocab'] is mostly <OoV>
    # (Pdb) print(sum(x != '<OoV>' for x in data['vocab']), 'out of', len(data['vocab']), ' is NOT <OoV>')
    # 27 out of 5835  is NOT <OoV>
    #
    # Debug>>>
    # (Pdb) model.mixture.weights.W.data.shape -> (11314, 20) (weights)
    # (Pdb) model.mixture.factors.W.data.shape -> (20, 300) (factors -> factor_vector)
    # (Pdb) model.sampler.W.data.shape -> (5837, 300) (word_vectors)
    # (Pdb) len(words) -> 5837 (vocab)
    if gpu_id >= 0:
        data = prepare_topics(cuda.to_gpu(model.mixture.weights.W.data).copy(),
                              cuda.to_gpu(model.mixture.factors.W.data).copy(),
                              cuda.to_gpu(model.sampler.W.data).copy(),
                              words, normalize = False)
    else:
        data = prepare_topics(cuda.to_cpu(model.mixture.weights.W.data).copy(),
                              cuda.to_cpu(model.mixture.factors.W.data).copy(),
                              cuda.to_cpu(model.sampler.W.data).copy(),
                              words, normalize = False)

    top_words = print_top_words_per_topic(data)

    if j % 100 == 0 and j > 100:
        coherence = topic_coherence(top_words)
        for j in range(n_topics):
            print(j, coherence[(j, 'cv')])
        kw = dict(top_words=top_words, coherence=coherence, epoch=epoch)
        progress[str(epoch)] = pickle.dumps(kw)

    data['doc_lengths'] = doc_lengths
    data['term_frequency'] = term_frequency
    np.savez('topics.pyldavis', **data)

    for d, f in utils.chunks(batchsize, doc_ids, flattened):
        t0 = time.time()

        # optimizer.zero_grads()
        model.cleargrads()

        l = model.fit_partial(d.copy(), f.copy(), update_only_docs=False)

        prior = model.prior()
        loss = clambda * prior * fraction
        loss.backward()
        optimizer.update()

        msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3e} "
               "P:{prior:1.3e} R:{rate:1.3e}")

        if gpu_id >= 0:
            prior.to_gpu()
            loss.to_gpu()
        else:
            prior.to_cpu()
            loss.to_cpu()

        t1 = time.time()
        dt = t1 - t0
        rate = batchsize / dt
        logs = dict(loss=float(l), epoch=epoch, j=j,
                    prior=float(prior.data), rate=rate)
        print(msg.format(**logs))
        j += 1

        if j % 12 == 0:
            logger.info('Word vectors grad: {}'.format(model.sampler.W.grad))
            logger.info('Document weights grad: {}'.format(model.mixture.weights.W.grad))
            logger.info('Topic matrix grad: {}'.format(model.mixture.factors.W.grad))

            params_log = ['{}/{},'.format(param.shape, param.name) for param in model.params()]
            logger.info('Parameters: {}'.format(params_log))

            dist1 = consine_distance(model.mixture.weights.W.data[0,:], model.mixture.weights.W.data[1,:])
            dist2 = consine_distance(model.mixture.weights.W.data[1,:], model.mixture.weights.W.data[2,:])
            logger.info('Doc/0-1 cosine: {}, /1-2 cosine: {}'.format(dist1, dist2))

            snapshot = prepare_topics(cuda.to_cpu(model.mixture.weights.W.data).copy(),
                                 cuda.to_cpu(model.mixture.factors.W.data).copy(),
                                 cuda.to_cpu(model.sampler.W.data).copy(),
                                 words, normalize = False)
            print_top_words_per_topic(snapshot)

    serializers.save_hdf5("lda2vec.hdf5", model)
