from tqdm import tqdm
from lda2vec import EmbedMixture
from lda2vec import dirichlet_likelihood
from lda2vec.utils import move

from chainer import Chain
import chainer.links as L
import chainer.functions as F

import numpy as np


class LDA2Vec(Chain):
    def __init__(self, n_documents=100, n_document_topics=10,
                 n_units=256, n_vocab=1000, dropout_ratio=0.5, train=True,
                 counts=None, n_samples=15, word_dropout_ratio=0.0,
                 power=0.75, temperature=1.0, vocab=None, docu_initialW=None):
        em = EmbedMixture(n_documents, n_document_topics, n_units,
                          dropout_ratio=dropout_ratio, temperature=temperature,
                          docu_initialW=docu_initialW)
        kwargs = {}
        kwargs['mixture'] = em
        # (Pdb) self.sampler.W.data.shape -> (4891, 300)
        kwargs['sampler'] = L.NegativeSampling(n_units, counts, n_samples,
                                               power=power)
        super(LDA2Vec, self).__init__(**kwargs)

        # note that sample.W.data will be loaded with pre-trained GoogleNews
        # word2vec data later in lda2vec_run.py
        rand = np.random.random(self.sampler.W.data.shape)
        self.sampler.W.data[:, :] = rand[:, :]

        self.n_units = n_units
        self.train = train
        self.dropout_ratio = dropout_ratio
        self.word_dropout_ratio = word_dropout_ratio
        self.n_samples = n_samples
        self.vocab = vocab

    def prior(self):
        dl1 = dirichlet_likelihood(self.mixture.weights)
        return dl1

    def fit_partial(self, rdoc_ids, rword_indices, window=5,
                    update_only_docs=False, word2vec_only=False):
        """ Compact indices of chunk words, from flattened
            (Pdb) len(rword_indices) -> 4096, batch size
            (Pdb) rword_indices.max() -> 4874, max word compact # in this chunk

            The belonged document ids of chunk words: 1660, from doc_ids
            (Pdb) len(rdoc_ids) -> 4096, batch size
            (Pdb) rdoc_ids.max() -> 1660, max doc id in this chunk
        """

        # Note that self.xp is module numpy. Function move uses following stmt
        # to convert both rdoc_ids and rword_indices as Chainer's Variable:
        # ---> yield Variable(xp.asarray(arg, dtype='float32'))
        #
        # so doc_ids and word_indices are just Variable wrapper of rdoc_ids
        # and rword_indices.
        # (Pdb) len(doc_ids.data) -> 4096
        # (Pdb) len(word_indices.data) -> 4096
        #
        # Note that doc_ids NOT IN USE
        doc_ids, word_indices = move(self.xp, rdoc_ids, rword_indices)

        # pivot_idx is Variable wrapper of rword_indices[window: -window]
        # (Pdb) len(pivot_idx.data) -> 4086, note that windows is 5
        pivot_idx = next(move(self.xp, rword_indices[window: -window]))

        # (Pdb) pivot.data.shape -> (4086, 300)
        # Again batchsize is 4096 while window size is (5, -5)
        pivot = F.embed_id(pivot_idx, self.sampler.W)

        # max word compact hash# < compacted vocabulary size (4891)
        assert pivot_idx.data.max() < self.sampler.W.shape[0]

        # Note that we meed to adjust word2vec from GoogleNews as we never
        # train word2vec using twenty_newgroups so that the context words prediction
        # not work well at the begining
        if update_only_docs:
            pivot.unchain_backward()

        # (Pdb) window -> 5
        # (Pdb) len(doc_at_pivot) -> 4086, 10 less than rdoc_ids
        # (Pdb) doc_at_pivot.max() -> 1660
        doc_at_pivot = rdoc_ids[window: -window]
        doc = self.mixture(next(move(self.xp, doc_at_pivot)),
                           update_only_docs=update_only_docs)
        if word2vec_only:
            doc.unchain_backward()
        loss = 0.0

        # (Pdb) start -> 5
        # (Pdb) rword_indices.shape[0] -> 4096
        # (Pdb) end -> 4091
        start, end = window, rword_indices.shape[0] - window

        # (Pdb) context.data.shape -> (4086, 300)
        context = (F.dropout(doc, self.dropout_ratio) +
                   F.dropout(pivot, self.dropout_ratio))

        # from -5 to 5, that is:
        # With given context vector (pivot wordvec + doc-topic_vec), predicts
        # each target word in the window frame
        for frame in tqdm(range(-window, window + 1)):
            # Skip predicting the current pivot
            if frame == 0:
                continue

            # Predict word given context and pivot word
            # The target starts before the pivot.
            #
            # Initial round:
            # (Pdb) start + frame -> 5 + -5 -> 0
            # (Pdb) end + frame -> 4091 + 5 -> 4086
            #
            # Word compact indices
            targetidx = rword_indices[start + frame: end + frame]

            # Word's document IDs
            doc_at_target = rdoc_ids[start + frame: end + frame]

            # Note that doc_at_pivot is rdoc_ids[window/5: -window/4091],
            # And      doc_at_target is rdoc_ids[0: 4086] in the starting round
            #
            # (Pdb) doc_is_same -> array([ True,  True,  True, ...,  True,  True,  True])
            # (Pdb) len(doc_is_same) -> 4086
            doc_is_same = doc_at_target == doc_at_pivot

            # (Pdb) rand -> array([0.7982769 , 0.12706805, 0.77982534, ..., 0.69266078])
            rand = np.random.uniform(0, 1, doc_is_same.shape[0])
            # (Pdb) mask -> array([ True,  True,  True, ...,  True,  True,  True])
            mask = (rand > self.word_dropout_ratio).astype('bool')
            # (Pdb) weight -> array([1, 1, 1, ..., 1, 1, 1], dtype=int32)
            weight = np.logical_and(doc_is_same, mask).astype('int32')

            # targetindex = target word indices
            # If weight is 1.0 then targetidx
            # If weight is 0.0 then -1, <SKIP>? => compact index 0
            # (Pdb) targetidx -> array([  28,    9, 2094, ...,   16, 1357,   16])
            #
            # Note that this is skip-gram, from pivot word -> target context words
            # See NegativeSampling below for ignore label -1.
            targetidx = targetidx * weight + -1 * (1 - weight)
            target, = move(self.xp, targetidx)

            # context, word_vec + docu-topic_vec, -> target words in context
            #
            # (Pdb) context.shape -> (4086, 300), dtype('float32')
            # (Pdb) weight.shape -> (4086,), dtype('int32')
            # (Pdb) targetidx.shape -> (4086,), dtype('int64')
            # (Pdb) target.shape -> (4086,), dtype('int32')
            # (Pdb) pivot_idx.shape -> (4086,), dtype('int32')
            # (Pdb) pivot.shape -> (4086, 300), dtype('float32')
            #
            # REF
            # self.sampler.__call__ =
            # negative_sampling.negative_sampling(
            #       x, t, self.W, self.sampler.sample, self.sample_size,
            #       reduce='sum')
            # here:
            # x (~chainer.Variable): Input of the weight matrix multiplication.
            # t (~chainer.Variable): Batch of ground truth labels.
            #
            # returns loss value
            #
            # Source (https://github.com/chainer/chainer/blob/v3.4.0/chainer/functions/loss/negative_sampling.py#L315)
            # NegativeSamplingFunction(function_node.FunctionNode):
            #       ignore_label = -1
            #       target as t -- self.sampler.W --> w
            #       context as x OP w --> loss
            # note that (Pdb) self.sampler.W.data.shape -> (4891, 300)
            loss = self.sampler(context, target)
            loss.backward()

            if update_only_docs:
                # Wipe out any gradient accumulation on word vectors
                # self.sampler.W.grad *= 0.0
                self.sampler.W.cleargrad()
            if word2vec_only and self.mixture.weights.W.grad is not None:
                assert self.mixture.weights.W.grad.min() == 0.0
                assert self.mixture.weights.W.grad.max() == 0.0
            if word2vec_only and self.mixture.factors.W.grad is not None:
                assert self.mixture.factors.W.grad.min() == 0.0
                assert self.mixture.factors.W.grad.max() == 0.0

        return loss.data
