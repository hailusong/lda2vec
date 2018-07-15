# lda2vec Network to Code Mapping
_[Chainer 3.4.0 Reference](https://docs.chainer.org/en/v3.4.0/reference/core/generated/chainer.Link.html#chainer.Link)_

## Static Analysis
### Topic Matrix
1. Shape: _**[n_topics, n_dim]**_
2. Flow:

  | Module | From | To | Notes |
  | - | - | - | - |
  | embed_mixture.py | L.Parameter(factors)) | | factors: Topic vector matrix. |
  | lda2vec_model.py | | kwargs['mixture'] | |
  | lda2vec_run.py | | model.mixture.factors.W.data | |
  | topics.py | | prepare_topics.factors | |

### Word Vector
Matrix of word vectors.<br>

1. Shape: _**[n_words, n_dim]**_
2. Flow:

  | Module | From | To | Notes |
  | - | - | - | - |
  | lda2vec_model.py | | rand = np.random.random(self.sampler.W.data.shape)<br>self.sampler.W.data[:, :] = rand[:, :] | Random initial? |
  | lda2vec_run.py | | model.sampler.W.data | |
  | topics.py | | prepare_topics.word_vectors | |

### Document Weight
an array of unnormalized log-odds of document-to-topic weights.<br>

1. Shape: _**[n_documents, n_topics]**_
2. Flow:

  | Module | From | To | Notes |
  | - | - | - | - |
  | embed_mixture.py | L.EmbedID(n_documents, n_topics) | | weights: unnormalized topic weights (:math:`c_j`).<br>To normalize these weights, use `F.softmax(weights)`.|
  | lda2vec_model.py | | kwargs['mixture'] |  |
  | lda2vec_run.py | | model.mixture.weights.W.data | |
  | topics.py | | prepare_topics.weights | |

## Diagram
![img](/lda2vec_network_publish_text.gif)
