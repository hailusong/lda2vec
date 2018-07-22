# Lda2Vec

## References
1. [Original readme](README-original.rst)
2. [Original blog](https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38&lambda=1&term=)
3. [LDA2vec: Word Embeddings in Topic Models](https://towardsdatascience.com/lda2vec-word-embeddings-in-topic-models-4ee3fc4b2843)

## Overview
1. Word2vec - skip-gram
  - Pivot word -> context words
  - Capture word-to-word relationship<br>
  <img src="https://multithreaded.stitchfix.com/assets/posts/2016-05-27-lda2vec/anim00.gif" width=50%/><br>
2. LDA<br>
  - Document vector -> words in document
  - Capture global relationship, not word-to-word relationship<br>
    <img src="https://multithreaded.stitchfix.com/assets/posts/2016-05-27-lda2vec/anim01.gif" width=50%/><br>
3. Lda2Vec<br>
  - Capture both global and local relationship at the same time<br>
    <img src="https://multithreaded.stitchfix.com/assets/posts/2016-05-27-lda2vec/anim02.gif" width=50%/><br>
  - The training is to minimize the following loss (more details [here](https://github.com/TropComplique/lda2vec-pytorch#losshttps://github.com/TropComplique/lda2vec-pytorch#loss))<br>
    <img src="https://github.com/TropComplique/lda2vec-pytorch/raw/master/loss.png" width=50%/><br>

## Implementations
1. [Original cemoody/lda2vec](https://github.com/cemoody/lda2vec)
2. [meereeum/lda2vec-tf](https://github.com/meereeum/lda2vec-tf)
3. [TropComplique/lda2vec-pytorch](https://github.com/TropComplique/lda2vec-pytorch)
4. [nateraw/Lda2vec-Tensorflow](https://github.com/nateraw/Lda2vec-Tensorflow)

## Problems

| Issue | In which implementation | Solutions |
| - | - | - |
| Topic matrix all similar | - nateraw/Lda2vec-Tensorflow | - better pre-processing to remove rare words<br>- do LEMMA |
| Negative Lda Loss | - nateraw/Lda2vec-Tensorflow<br>- meereeum/lda2vec-tf | - positive it |
| - Usually a lot of found topics are a total mess.<br>- the algorithm is prone to poor local minima.<br>- it greatly depends on values of initial topic assignments | - TropComplique/lda2vec-pytorc | - do LEMMA<br>- use vanilla LDA to initialize document's topic assignments<br> - use temperature to smoothen the initialization in the hope that lda2vec will have a chance to find better topic assignments.<br>- remove BOTH **rare** and **frequent** words |

### Trail and Error #1
1. Use LEMMA
  - Convert '-PRON-' to '\<SKIP\>'
2. When doing NCE,
  - replace all '\<SKIP\>' and OOV wtih chainer.NegativeSampling.ignore_label (-1)

### Trail and Error #2
1. We may request too much from a NCE model which is designed to build word-to-word relationship.
   Now we expect the model do build all 3 relationship from one loss function:
  - **word-to-word**: word vectors weights, in word vector space
  - **document-to-topic**: document topic weights, in document topic proportion
  - **topic-to-word**: topic weights in word vector space
2. Given that we already have word-to-word relationship (loss minimized by GoogleNews word2vec),
  - simply adding **word vec + context vec** and do the loss backward will naturally ...
  - make the model push all topics in word vector space to the same point so that ...
  - **context vec** won't intervene the original **word vec** setup, as such ...
  - the model can enjoy the minimized loss already introduced by GoogleNews word2vec
  - also if all topic vecs moved to the same point, the document weight proportion no more matter  ...
  - as the matmul result will be the same
3. The better approach could be
  - use fasttext to build **word vec** data from corpus (not from GoogleNews word2vec) so that ...
  - we have a better minimized loss on word to word relationship
  - do vanilla LDA on corpus, that is ...
  - not to add **word vec** to **context vec**, just use **context vec** to globally predict ...
  - all document words and apply loss backward which will ...
  - update **document topic weights** and **topic weights**

### Next Step
1. [Lda2vec intriguing though not _**very impressive**_ at the moment](http://nlpx.net/archives/330)
  - Current lda2vec implementation to produce good output
  - But it is not significantly better than the output of pure LDA
2. **Thought #1** is to use word2vec to build word-to-word relationship, and then
  - Run **k-means** to generate topic clusters in w2v space, then
  - Modify **lda2vec** to only back-propagation on **document topic weights**
  - Similar approach [LSA + k-means](Latent Semantic Analysis with K-Means)
3. **Thought #2** is can we somehow make word2vec can also build global relationship?
  - Like document-to-word, additional-attribute-to-word
  - add document id and other attributes as _**prefix**_ to words, or
4. Other though include
  - Can we train the model on tagged document-topic corpus and then use the model to predict topics of a new document?
  - The answer could be simple **text classification** using _**supervised fasttext**_
