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
_Matrix of word vectors._<br>

1. Shape: _**[n_words, n_dim]**_
2. Flow:
  | Module | From | To | Notes |
  | - | - | - | - |
  | lda2vec_model.py | | rand = np.random.random(self.sampler.W.data.shape)<br>self.sampler.W.data[:, :] = rand[:, :] | Random initial? |
  | lda2vec_run.py | | model.sampler.W.data | |
  | topics.py | | prepare_topics.word_vectors | |

### Document Weight
_An array of unnormalized log-odds of document-to-topic weights._<br>

1. Shape: _**[n_documents, n_topics]**_
2. Flow:
  | Module | From | To | Notes |
  | - | - | - | - |
  | embed_mixture.py | L.EmbedID(n_documents, n_topics) | | weights: unnormalized topic weights (:math:`c_j`).<br>To normalize these weights, use `F.softmax(weights)`.|
  | lda2vec_model.py | | kwargs['mixture'] |  |
  | lda2vec_run.py | | model.mixture.weights.W.data | |
  | topics.py | | prepare_topics.weights | |

### Pivot Word
_Extract **pairs of pivot and target words** that occur in the **moving window** that scans across the **corpus**_.<br>
_For every pair, **pivot** is used to predict **the nearby arget word**._<br>

1. To **word vector** flow:
  | Module | From | To | Notes |
  | - | - | - | - |
  | lda2vec_model.py:fit_partial | pivot_idx = next(move(self.xp, rword_indices[window: -window]))<br>pivot = F.[embed_id](https://docs.chainer.org/en/v3.4.0/reference/generated/chainer.functions.embed_id.html?highlight=embed_id#chainer.functions.embed_id)(pivot_idx, self.sampler.W) | | |

### Document proportion
_The **document weights** are **softmax** transformed weights to yield the **document proportions**_.<br>

1. Flow:
  | Module | From | To | Notes |
  | - | - | - | - |
  | embed_mixture.py:__call__ | self.proportions(doc_ids, softmax=True) | | sums to 100% and indicates the topic proportions of a single document |

## Model Analysis
| Module | Input #1 | Input #2 | Op | Output | Notes |
| - | - | - | - | - | - |
| lda2vec_model.py:fit_partial | **Word vector**<br>F.dropout(pivot, self.dropout_ratio) | **Document vector**<br>F.dropout(doc, self.dropout_ratio) | + | Context vector - **context** | |
| embed_mixture.py:__call__ | **Document proportion** | **Topic matrix** | F.matmul | Document vector | |

## Diagram
![img](/lda2vec_network_publish_text.gif)
