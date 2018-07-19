# Lda2Vec

## References
1. [Original blog](https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38&lambda=1&term=)
2. [LDA2vec: Word Embeddings in Topic Models](https://towardsdatascience.com/lda2vec-word-embeddings-in-topic-models-4ee3fc4b2843)

## Overview
1. Word2vec - skip-gram
  - Pivot word -> context words<br>
  - Capture word-to-word relationship
  <img src="https://multithreaded.stitchfix.com/assets/posts/2016-05-27-lda2vec/anim00.gif" width=50%/>
2. LDA
  - Document vector -> words in document
  - Capture global relationship, not word-to-word relationship<br>
    <img src="https://multithreaded.stitchfix.com/assets/posts/2016-05-27-lda2vec/anim01.gif" width=50%/>
3. Lda2Vec
  - Capture both global and local relationship at the same time<br>
    <img src="https://multithreaded.stitchfix.com/assets/posts/2016-05-27-lda2vec/anim02.gif" width=50%/>

## Implementations
1. [Original cemoody/lda2vec](https://github.com/cemoody/lda2vec)
2. [meereeum/lda2vec-tf](https://github.com/meereeum/lda2vec-tf)
3. [TropComplique/lda2vec-pytorch](https://github.com/TropComplique/lda2vec-pytorch)
4. [nateraw/Lda2vec-Tensorflow](https://github.com/nateraw/Lda2vec-Tensorflow)

## Problems

| Issue | In which implementation | Solutions |
| - | - | - |
| Topic matrix all similar | - nateraw/Lda2vec-Tensorflow | - better pre-processing to remove rare words<br>- do LEMMA |
| Negative Lda Loss | - nateraw/Lda2vec-Tensorflow<br>- meereeum/lda2vec-tf | positive it |
| - Usually a lot of found topics are a total mess.<br>- the algorithm is prone to poor local minima.<br>- it greatly depends on values of initial topic assignments | TropComplique/lda2vec-pytorc | - do LEMMA<br>- Use vanilla LDA to initialize document's topic assignments<br> - use temperature to smoothen the initialization in the hope that lda2vec will have a chance to find better topic assignments.<br>- remove BOTH **rare** and **frequent** words |
