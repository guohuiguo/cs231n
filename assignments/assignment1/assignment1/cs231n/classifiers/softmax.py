from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    和我一开始想的矩阵乘法的方向刚好是相反的 🤔

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1] # 3073*class
    num_train = X.shape[0] # num*3073
    for i in range(num_train):
        scores = X[i].dot(W) #每个data在每个class上得到的scores


        # compute the probabilities in numerically stable way
        # scores -= np.max(scores) 防止指数溢出 确实👍
        scores -= np.max(scores) #这个这么理解，为什么要这样做🤔，没什么影响相当于上下除以exp(max(scores))
        p = np.exp(scores)
        p /= p.sum()  # normalize 归一化，我能理解
        logp = np.log(p)
        
        dW[:,y[i]] -= X[i]
        tmp = np.zeros_like(W)
        for c in range(num_classes):
          tmp[:,c] = p[c] * X[i] # 这个用一个行向量和一个列向量的相乘就可以得到这个每一列单独乘以对应东西的矩阵，无需再升维了
        dW += tmp


        
        loss -= logp[y[i]]  # negative log probability is the loss
        #对应y[i]的label的loss

        #deepseek's simple code
        '''
        # 更简洁的等价实现 和我的没有什么区别！
        dW_i = np.zeros_like(W)
        for c in range(num_classes):
            dW_i[:, c] = p[c] * X[i]  # 使用 X[i] 替代 X[i].T
            
        # 真实类别特殊处理
        dW_i[:, y[i]] -= X[i]

        dW += dW_i
        '''





    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2*reg*W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0] # num*3073
    num_classes = W.shape[1] # 3073*class
    num_d = W.shape[0]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    # 更简洁的等价实现 和我的没有什么区别！
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims = True)
    p = np.exp(scores)
    p /= np.sum(p,axis=1,keepdims=True)
    logp = np.log(p)
    loss -= np.sum(logp[np.arange(num_train),y]) 

    Y_onehot = np.zeros_like(p)
    Y_onehot[np.arange(num_train),y] = 1

    dW = X.T.dot(p-Y_onehot) 
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    loss = loss/ num_train + reg * np.sum(W*W) 
    dW = dW / num_train + 2 * reg *W

    return loss, dW








































    
