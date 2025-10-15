from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *
from cs231n import layers  



class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32, # D
        hidden_dim=100,  # what it is隐藏层维度为H
        num_classes=10, # C
        weight_scale=1e-3, #parameters
        reg=0.0, #parameters
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer # the num of the layber?
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        #self how to use?
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases（这个东西没有显式见到） should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        # w1 b1 w2 b2
        ############################################################################
        self.params['W1'] = np.random.normal(0, weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)
        #self.w1 = np.random.normal(0, weight_scale, size=(input_dim, hidden_dim))
        #self.b1 = np.zeros(1,hidden_dim)

        #self.w2 = np.random.normal(0, weight_scale, size=(hidden_dim, num_classes))
        #self.b2 = np.zeros(1,num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        num_train = X.shape[0]
        w1, b1 = self.params['W1'], self.params['b1']
        w2, b2 = self.params['W2'], self.params['b2']
        reg = self.reg
        scores = None
        scores = np.zeros((num_train, self.num_classes))
        scores_1 = np.zeros((num_train,self.input_dim))
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # first affine scores_1
        X_flat = X.reshape(num_train, -1)
        scores_1 = X_flat.dot(w1) + b1
        
        # ReLU : scores_2
        scores_2 = np.maximum(0,scores_1)

        # second affine
        scores = scores_2.dot(w2) + b2
        # softmax 但是关于。 没有y_label是没有办法计算softmax的
        # 再看要求输出为 scores: Array of shape (N, C)。所以我们直接前三层就得到了


        # 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # softmax算法 最复杂的一块 问题来了，dw1 dx1 dw2 dx2 db1 db2是不是都要求呀🤔

        scores -= np.max(scores, axis=1, keepdims = True)
        p = np.exp(scores)
        p /= np.sum(p,axis=1,keepdims=True)
        logp = np.log(p)
        loss -= np.sum(logp[np.arange(num_train),y]) 

        # onehot
        Y_onehot = np.zeros_like(p)
        Y_onehot[np.arange(num_train),y] = 1
        
        # Init
        dw2 = np.zeros_like(w2)
        db2 = np.zeros_like(b2)
        # 这里也要平均一下
        dscores = (p-Y_onehot) / num_train

        # cal & don't forget L2 regularization
        dw2 = scores_2.T.dot(dscores)
        db2 = np.sum(dscores, axis=0)

        grads['W2'] = dw2 + reg * w2
        grads['b2'] = db2

        # 计算loss关于 w1 b1的梯度，这个由于之间有很多层，就很麻烦了
        
        # Init
        dw1 = np.zeros_like(self.params['W1'])
        db1 = np.zeros_like(self.params['b1'])

        # 计算
        dw1 = X_flat.T.dot((dscores.dot(w2.T))*(scores_1>0))
        db1 = np.sum((dscores.dot(w2.T))*(scores_1>0), axis=0)

        grads['W1'] = dw1 + reg * w1
        grads['b1'] = db1
        

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        loss = loss / num_train + 0.5 * reg * np.sum(w2 * w2) + 0.5*reg*np.sum(w1*w1)

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        layers_dims=[input_dim]+hidden_dims+[num_classes]



        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################

        for i in range(1, self.num_layers+1):
          self.params[f"W{i}"]=np.random.normal(loc=0,scale=weight_scale,size=(layers_dims[i-1], layers_dims[i])).astype(self.dtype)
          self.params[f'b{i}'] = np.zeros(layers_dims[i], dtype=self.dtype)

          if self.normalization == "batchnorm" and i <= self.num_layers - 1:  # 明白归一化怎么处理的，但是不知道函数怎么写
                # 缩放参数gamma初始化为1
                self.params[f'gamma{i}'] = np.ones(layers_dims[i], dtype=self.dtype)
                # 偏移参数beta初始化为0
                self.params[f'beta{i}'] = np.zeros(layers_dims[i], dtype=self.dtype)



        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        cache = {}  # 键：层标识（如'affine1'），值：该层的中间变量
        N = X.shape[0]  # 样本数量

        # 输入数据展平（适应全连接层，形状从(N, d1, d2, ...)变为(N, D)）
        out = X.reshape(N, -1)

        # 前(L-1)层：affine -> [batchnorm] -> relu -> [dropout]
        for i in range(1, self.num_layers):  # i=1到L-1（L为总层数）
            # 1. 仿射层（affine: y = Wx + b）
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            out, cache[f'affine{i}'] = layers.affine_forward(out, W, b)  # 调用仿射层前向函数

            # 2. 批量归一化（若启用）
            if self.normalization == "batchnorm":
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']
                bn_param = self.bn_params[i-1]  # 第i层对应第i-1个bn参数
                out, cache[f'bn{i}'] = layers.batchnorm_forward(out, gamma, beta, bn_param)  # 调用BN前向函数

            # 3. ReLU激活函数
            out, cache[f'relu{i}'] = layers.relu_forward(out)  # 调用ReLU前向函数

            # 4. Dropout（若启用）
            if self.use_dropout:
                out, cache[f'dropout{i}'] = layers.dropout_forward(out, self.dropout_param)  # 调用dropout前向函数

        # 最后一层（第L层）：仅仿射层（无激活/归一化，直接输出scores）
        W_final = self.params[f'W{self.num_layers}']
        b_final = self.params[f'b{self.num_layers}']
        scores, cache[f'affine{self.num_layers}'] = layers.affine_forward(out, W_final, b_final)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}


        # 1. 计算softmax数据损失和输出层梯度（dscores）
        data_loss, dscores = layers.softmax_loss(scores, y)  # 调用softmax损失函数

        # 2. 计算L2正则化损失（仅对权重W，不含偏置和BN的gamma/beta）
        reg_loss = 0.0
        for i in range(1, self.num_layers + 1):
            W = self.params[f'W{i}']
            reg_loss += 0.5 * self.reg * np.sum(W **2)  # 含0.5因子，简化梯度计算
        loss = data_loss + reg_loss  # 总损失 = 数据损失 + 正则化损失

        # 3. 反向传播：从输出层向输入层回传梯度

        # 3.1 最后一层（第L层）仿射层的反向传播
        i = self.num_layers
        dx, dW, db = layers.affine_backward(dscores, cache[f'affine{i}'])  # 调用仿射层反向函数
        grads[f'W{i}'] = dW + self.reg * self.params[f'W{i}']  # 加上正则化梯度
        grads[f'b{i}'] = db

        # 3.2 前(L-1)层的反向传播（从L-1层到1层）
        for i in reversed(range(1, self.num_layers)):  # 逆序遍历：L-1 → 1
            # a. Dropout反向（若启用）
            if self.use_dropout:
                dx = layers.dropout_backward(dx, cache[f'dropout{i}'])  # 调用dropout反向函数

            # b. ReLU反向
            dx = layers.relu_backward(dx, cache[f'relu{i}'])  # 调用ReLU反向函数

            # c. 批量归一化反向（若启用）
            if self.normalization == "batchnorm":
                dx, dgamma, dbeta = layers.batchnorm_backward(dx, cache[f'bn{i}'])  # 调用BN反向函数
                grads[f'gamma{i}'] = dgamma
                grads[f'beta{i}'] = dbeta

            # d. 仿射层反向
            dx, dW, db = layers.affine_backward(dx, cache[f'affine{i}'])  # 调用仿射层反向函数
            grads[f'W{i}'] = dW + self.reg * self.params[f'W{i}']  # 加上正则化梯度
            grads[f'b{i}'] = db
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True