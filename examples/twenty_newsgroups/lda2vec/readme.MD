# lda2vec Network to Code Mapping
_[Chainer 3.4.0 Reference](https://docs.chainer.org/en/v3.4.0/reference/core/generated/chainer.Link.html#chainer.Link)_<br>
- [optimizer.update()](https://docs.chainer.org/en/v3.4.0/reference/core/generated/chainer.Optimizer.html#chainer.Optimizer)
- [Variable loss.backward()](https://docs.chainer.org/en/v3.4.0/reference/core/generated/chainer.Variable.html?highlight=backward#chainer.Variable.backward)
  * <b>retain_grad</b>: In most cases of training some models, the purpose of backprop is to compute gradients of parameters, not of all variables, and therefore it is recommended to set this flag <b>False</b>

## LDA2Vec Model Constructor Params

| Name | Value | Notes |
| - | - | - |
| n_documents | 11314 | document # in data set |
| n_document_topics | 20 | topic # |
| n_units | 300 | embedding dimensions |
| n_vocab | 5837 | word count in vocabulary |
| dropout_ratio | 0.5 | |
| train | True | |
| n_samples | 15 | negative sampler size |

## Static Analysis
### Topic matrix
_Each topic has a distributed representation that lives in the same space as word vectors._<br>

1. Shape: _**[n_topics, n_dim]**_ => (20, 300)
2. Flow:

  | Module | From ... To ... | Notes |
  | - | - | - |
  | embed_mixture.py | factors = \_orthogonal_matrix((n_topics, n_dim)).astype('float32')<br>factors /= np.sqrt(n_topics + n_dim)<br>mixture.factors = L.[Parameter](https://docs.chainer.org/en/v3.4.0/reference/generated/chainer.links.Parameter.html)(factors)) | factors: Topic vector matrix. |
  | lda2vec_model.py | kwargs['mixture'] | |
  | lda2vec_run.py | model.mixture.factors.W.data | |
  | topics.py | prepare_topics.factors | |

### Word vector
_Matrix of word vectors. Each pivot word is represented with a fixed-length dense distributed representation vector._<br>
_These has ALL of word2vec's familiar properties_.<br>

1. Shape: _**[n_words, n_dim]**_ => (5837, 300)
2. Flow:

  | Module | From ... To ... | Notes |
  | - | - | - |
  | lda2vec_model.py | rand = np.random.random(self.sampler.W.data.shape)<br>self.sampler.W.data[:, :] = rand[:, :] | Random initial? |
  | lda2vec_run.py | model.sampler.W.data | |
  | topics.py | prepare_topics.word_vectors | |

### Document weight
_An array of unnormalized log-odds of document-to-topic weights._<br>

1. Shape: _**[n_documents, n_topics]**_ => (11314, 20)
2. Flow:

  | Module | From ... To ... | Notes |
  | - | - |  - |
  | embed_mixture.py | mixture.weights = L.[EmbedID](https://docs.chainer.org/en/v3.4.0/reference/generated/chainer.links.EmbedID.html?highlight=embedid)(n_documents, n_topics) | weights: unnormalized topic weights (:math:`c_j`).<br>To normalize these weights, use `F.softmax(weights)`.|
  | lda2vec_model.py | kwargs['mixture'] |  |
  | lda2vec_run.py | model.mixture.weights.W.data | |
  | topics.py | prepare_topics.weights | |

### Pivot word
_Extract **pairs of pivot and target words** that occur in the **moving window** that scans across the **corpus**_.<br>
_For every pair, **pivot** is used to predict **the nearby arget word**._<br>

1. To **word vector** flow:

  | Module | From ... To ... | Notes |
  | - | - | - |
  | lda2vec_model.py:fit_partial | pivot_idx = next(move(self.xp, rword_indices[window: -window]))<br>pivot = F.[embed_id](https://docs.chainer.org/en/v3.4.0/reference/generated/chainer.functions.embed_id.html?highlight=embed_id#chainer.functions.embed_id)(pivot_idx, self.sampler.W) | |

### Document proportion
_The **document weights** are **softmax** transformed weights to yield the **document proportions**_.<br>

1. Flow:

  | Module | From ... To ... | Notes |
  | - | - | - |
  | embed_mixture.py:\__call__ | self.proportions(doc_ids, softmax=True) | sums to 100% and indicates the topic proportions of a single document |

### Target word

1. Flow:

  | Module | From ... To ... | Notes |
  | - | - | - |
  | lda2vec_model.py:fit_partial | _see below_ |  |

  ```
  targetidx = rword_indices[start + frame: end + frame]
  doc_at_target = rdoc_ids[start + frame: end + frame]
  doc_is_same = doc_at_target == doc_at_pivot
  rand = np.random.uniform(0, 1, doc_is_same.shape[0])
  mask = (rand > self.word_dropout_ratio).astype('bool')
  weight = np.logical_and(doc_is_same, mask).astype('int32')
  # If weight is 1.0 then targetidx
  # If weight is 0.0 then -1
  targetidx = targetidx * weight + -1 * (1 - weight)
  target, = move(self.xp, targetidx)
  ```

## Model Analysis

| Module | Input #1 | Input #2 | Op | Output | Notes |
| - | - | - | - | - | - |
| lda2vec_model.py:<br>fit_partial() | **Word vector**<br>F.dropout(pivot, self.dropout_ratio) | **Document vector**<br>F.dropout(doc, self.dropout_ratio) | + | Context vector - **context** | |
| embed_mixture.py:<br>\__call__() | **Document proportion** | **Topic matrix** | F.matmul | Document vector | |
| lda2vec_model.py:<br>fit_partial() | **Context vector** | **Target word** | [L.NegativeSampling](https://docs.chainer.org/en/v3.4.0/reference/generated/chainer.links.NegativeSampling.html?highlight=negativesampling) | _loss value_ |
| lda2vec_run.py | **Document weight** | | dirichlet_likelihood<br>_lda2vec_model.py:prior()_ | _loss value_ | Sparse document proportions.<br>dirichlet_likelihood calculates the log likelihood of the observed topic proportions. |

### Loss functions

| Loss | Between what | and what | Where | Notes | Logging |
| - | - | - | - | - | - |
| [L.NegativeSampling](https://docs.chainer.org/en/v3.4.0/reference/generated/chainer.links.NegativeSampling.html?highlight=negativesampling) | _**context vector**_<br>a.k.a. _pivot word vector + document vector_ | _**target word**_ | lda2vec_model.py:fit_partial() | For each mini-batch...<br>&nbsp;&nbsp;_do loss.backward() in each moving frame window_ | _**L**_ |
| dirichlet_likelihood | _**mixture.weights**_<br>a.k.a. _document weights_ | | lda2vec_run.py | Do loss.backward() in each mini-batch | _**P**_ |

## Diagram
![img](/lda2vec_network_publish_text.gif)
