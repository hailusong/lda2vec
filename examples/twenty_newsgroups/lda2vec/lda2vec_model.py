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
        # (Pdb) n_units -> 300, embedding dimensions
        # (Pdb) counts -> array([ 0,  0,  0, ..., 30, 30, 29], dtype=int32)
        # (Pdb) counts.shape -> (4891,)
        # (Pdb) len(vocab) -> 4891
        # (Pdb) vocab[0] -> '<SKIP>', vocab[1] -> 'out_of_vocabulary',  vocab[2] -> '-PRON-'
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
                    update_only_docs=False,
                    word2vec_only=False,
                    update_only_docs_topics=False):
        """ Compact indices of chunk words, from flattened
            (Pdb) len(rword_indices) -> 4096, batch size
            (Pdb) rword_indices.max() -> 4874, max word compact # in this chunk

            The belonged document ids of chunk words: 1660, from doc_ids
            (Pdb) len(rdoc_ids) -> 4096, batch size
            (Pdb) rdoc_ids.max() -> 1660, max doc id in this chunk
        """

        if update_only_docs_topics:
            update_only_docs = False

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
        if update_only_docs or update_only_docs_topics:
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
        if not update_only_docs_topics:
            context = (F.dropout(doc, self.dropout_ratio) +
                       F.dropout(pivot, self.dropout_ratio))
        else:
            context = F.dropout(doc, self.dropout_ratio)

        # from -5 to 5, that is:
        # With given context vector (pivot wordvec + doc-topic_vec), predicts
        # each target word in the window frame.
        # Note that we do this for all words in the whole batch size.
        for frame in tqdm(range(-window, window + 1)):
            # Skip predicting the current pivot
            if frame == 0 and not update_only_docs_topics:
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

            # Since we flatten everything: all words from all different documents
            # now in one array, we need to make sure we only predict words in the
            # same document.
            #
            # Note that doc_at_pivot is rdoc_ids[window/5: -window/4091],
            # And      doc_at_target is rdoc_ids[0: 4086] in the starting round
            #
            # (Pdb) doc_is_same -> array([ True,  True,  True, ...,  True,  True,  True])
            # (Pdb) len(doc_is_same) -> 4086
            doc_is_same = doc_at_target == doc_at_pivot

            # Generate <SKIP>, OOV mask
            mask_SKIP = targetidx != np.array([0])
            mask_OOV = targetidx != np.array([1])
            assert True in mask_SKIP and True in mask_OOV

            # Generate drop-out mask
            # (Pdb) rand -> array([0.7982769 , 0.12706805, 0.77982534, ..., 0.69266078])
            rand = np.random.uniform(0, 1, doc_is_same.shape[0])
            # (Pdb) mask -> array([ True,  True,  True, ...,  True,  True,  True])
            mask = (rand > self.word_dropout_ratio).astype('bool')

            # (Pdb) weight -> array([1, 1, 1, ..., 1, 1, 1], dtype=int32)
            weight = np.logical_and(doc_is_same, mask)
            weight = np.logical_and(weight, mask_SKIP)
            weight = np.logical_and(weight, mask_OOV).astype('int32')

            # targetindex = target word indices
            # If weight is 1.0 then targetidx
            # If weight is 0.0 then -1, <SKIP>? => compact index 0
            # (Pdb) targetidx -> array([  28,    9, 2094, ...,   16, 1357,   16])
            #
            # Note that this is skip-gram, from pivot word -> target context words
            # See NegativeSampling below for ignore label -1.
            chainer_nce_ignore_label = -1
            targetidx = targetidx * weight + chainer_nce_ignore_label * (1 - weight)
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
            # context -> x (~chainer.Variable): Input of the weight matrix multiplication.
            # target -> t (~chainer.Variable): Batch of ground truth labels.
            # GoogleNews Embedding -> self.sampler.W.data
            # L.NegativeSampling -> sampler
            #
            # returns loss value, sum of all losses on the whole batchsize data.
            #
            # Source (https://github.com/chainer/chainer/blob/v3.4.0/chainer/functions/loss/negative_sampling.py#L315)
            # NegativeSamplingFunction(function_node.FunctionNode):
            #       ignore_label = -1
            #       target as t -- self.sampler.W --> w
            #       context as x OP w --> loss
            # note that (Pdb) self.sampler.W.data.shape -> (4891, 300)
            #
            # DEBUG
            # b chainer/functions/loss/negative_sampling.py:48
            loss = self.sampler(context, target)
            loss.backward()

            if update_only_docs or update_only_docs_topics:
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
