# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import pickle
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import os
os.sys.path.append('../../../')

from lda2vec import preprocess, Corpus
from lda2vec.logging import logger
from spacy.attrs import LEMMA


# Fetch data
remove = ('headers', 'footers', 'quotes')
texts = fetch_20newsgroups(subset='train', remove=remove).data
# Remove tokens with these substrings
bad = set(["ax>", '`@("', '---', '===', '~~~', '......', '=-=-=', '_____', '^^^', '|||', '\\\\', '////', '%', '@', ':', '$'])


def clean(line):
    return ' '.join(w for w in line.split() if not any(t in w.lower() for t in bad))

# Preprocess data
max_length = 10000   # Limit of 10k words per document
# Convert to unicode (spaCy only works with unicode)
# texts = [unicode(clean(d)) for d in texts]
texts = [clean(d) for d in texts]

# vocab - dictionary where keys are the loose index, and values are the word string.
# (Pdb) len(vocab) -> 74179
# (Pdb) type(vocab) -> <class 'dict'>
#
# tokens, 2D array, one row per document, columns are long hash# of words in the sequence of
# the occuring in the document (max words in one document is 10K)
# (Pdb) tokens.shape -> (11314, 10000)
tokens, vocab = preprocess.tokenize(texts, max_length, merge=False, attr=LEMMA,
                                    n_threads=8)
corpus = Corpus()
# Make a ranked list of rare vs frequent words
corpus.update_word_count(tokens)
corpus.finalize()

# The tokenization uses spaCy indices, and so may have gaps
# between indices for words that aren't present in our dataset.
# This will build the 2D array, one row per document, columns are word compact hash#
# in the sequence of the occuring in the document (max words in one document is 10K)
#
# (Pdb) len(tokens) -> 11314
# (Pdb) compact.shape -> (11314, 10000)
# (Pdb) compact.max() -> 74179
compact = corpus.to_compact(tokens)
logger.info('compact inspection: {},{},{}'.format(compact.max(), compact.min(), compact.dtype))

# Remove extremely rare words. All rare words in the pruned will be replaced with
# OOV.
# (Pdb) pruned.shape -> (11314, 10000)
# (Pdb) pruned.max() -> 4888
pruned = corpus.filter_count(compact, min_count=30)
logger.info('pruned inspection: {},{},{}'.format(pruned.max(), pruned.min(), pruned.dtype))
pruned = pruned.astype(np.int64, casting='unsafe')

# Convert the compactified arrays into bag of words arrays.
# Looks like after pruned the number of words is now 4891.
# One row per document, columns are word frequency in the sequence of word compact #
# (Pdb) bow.shape -> (11314, 4891)
# (Pdb) bow.max() -> 10000, that likely belong to padding words like OOV or SKIP
# (Pdb) bow
# array([[9892,    7,    7, ...,    0,    0,    0],
#        [9898,    4,    5, ...,    0,    0,    0],
#        [9649,   16,   11, ...,    0,    0,    0],
#        ...,
#        [9875,    7,    7, ...,    0,    0,    0],
#        [9931,    6,    5, ...,    0,    0,    0],
#        [9941,    8,    4, ...,    0,    0,    0]])
bow = corpus.compact_to_bow(pruned)
logger.info('bow inspection: {},{},{}'.format(bow.max(), bow.min(), bow.dtype))

# Words tend to have power law frequency, so selectively
# downsample the most prevalent words. All too-frequent words replaced
# with <SKIP> - currently not in use probabily because hord to determine
# safely how frequent is considered as too frequent (STOP_WORDS?).
# (Pdb) !clean.shape -> (11314, 10000)
clean = corpus.subsample_frequent(pruned)

# Now flatten a 2D array of document per row and word position
# per column to a 1D array of words. This will also remove skips
# and OoV words.
# (Pdb) flattened.shape -> (2112661,), all word IDs of all documents in 1D array
# (Pdb) flattened -> array([  6, 591,  30, ...,  43,  43,  43])
# (Pdb) doc_ids.shape -> (2112661,), all word's document IDs in 1D array
# (Pdb) doc_ids -> array([    0,     0,     0, ..., 11313, 11313, 11313])
doc_ids = np.arange(pruned.shape[0])
flattened, (doc_ids,) = corpus.compact_to_flat(pruned, doc_ids)
assert flattened.min() >= 0

# Fill in the pretrained word vectors
n_dim = 300
fn_wordvc = 'GoogleNews-vectors-negative300.bin'
vectors, s, f = corpus.compact_word_vectors(vocab, filename=fn_wordvc)

# Save all of the preprocessed files
pickle.dump(vocab, open('vocab.pkl', 'wb'))
pickle.dump(corpus, open('corpus.pkl', 'wb'))

# (Pdb) flattened.shape -> (2125958,)
np.save("flattened", flattened)

# (Pdb) doc_ids.shape -> (2125958,)
np.save("doc_ids", doc_ids)

# [documents, words], words in the sequence of occuring in each document
# note that rare words have been replaced with OOV
# (Pdb) pruned.shape -> (11314, 10000)
np.save("pruned", pruned)

# [documents, bag-of-words], words in the sequence of compact hash# 1 -> 4891
# (Pdb) bow.shape -> (11314, 4891)
np.save("bow", bow)

# [our vocabulary, embedding dimensions (from GoogleNews embedding)]
# (Pdb) vectors.shape -> (74280, 300)
np.save("vectors", vectors)
