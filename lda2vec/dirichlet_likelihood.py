import chainer.functions as F
from chainer import Variable
from lda2vec.logging import logger


def dirichlet_likelihood(weights, alpha=None):
    """ Calculate the log likelihood of the observed topic proportions.
    A negative likelihood is more likely than a negative likelihood.

    Args:
        weights (chainer.Variable): Unnormalized weight vector. The vector
            will be passed through a softmax function that will map the input
            onto a probability simplex.
        alpha (float): The Dirichlet concentration parameter. Alpha
            greater than 1.0 results in very dense topic weights such
            that each document belongs to many topics. Alpha < 1.0 results
            in sparser topic weights. The default is to set alpha to
            1.0 / n_topics, effectively enforcing the prior belief that a
            document belong to very topics at once.

    Returns:
        ~chainer.Variable: Output loss variable.
    """
    if type(weights) is Variable:
        n_topics = weights.data.shape[1]
    else:
        n_topics = weights.W.data.shape[1]
    # logger.info('dirichlet_likelihood on topics of {}'.format(n_topics))

    if alpha is None:
        alpha = 1.0 / n_topics

    if type(weights) is Variable:
        log_proportions = F.log_softmax(weights)
    else:
        log_proportions = F.log_softmax(weights.W)

    # positive
    loss = (alpha - 1.0) * log_proportions
    # negative
    # return -F.sum(loss)

    return F.sum(loss)
