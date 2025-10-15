from builtins import range
import numpy as np
from numpy.random import sample


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    out = x.reshape(x.shape[0],-1).dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0],-1).T.dot(dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    dx=dout*(x>0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    loss=0
    dx = np.zeros_like(x)
    num_train = x.shape[0]

    shifted_x = x - np.max(x, axis=1, keepdims=True)
    p = np.exp(shifted_x)
    p /= np.sum(p,axis=1,keepdims=True)
    logp = np.log(p)
    loss -= np.sum(logp[np.arange(x.shape[0]),y]) 
    # 当j等于y_i的地方，还是用onehot，统一化标准。
    Y_onehot = np.zeros_like(p)
    Y_onehot[np.arange(num_train),y] = 1 
    dx = (p - Y_onehot) / num_train
    loss = loss / num_train
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        sample_mean = np.mean(x,axis=0)
        sample_var = np.sum((x-sample_mean)**2,axis=0) / N
        x_norm = (x-sample_mean)/np.sqrt(sample_var+eps)
        out = gamma*x_norm+beta
        #running的mean和var是维护的，在训练中不参与计算。我们只需要算out关于x的梯度
        running_mean=momentum*running_mean+(1-momentum)*sample_mean
        running_var=momentum*running_var+(1-momentum)*sample_var
        # cache 都需要包括什么？
        cache = (x, gamma, beta, sample_mean, sample_var, x_norm, eps)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_norm = (x-running_mean)/np.sqrt(running_var+eps)
        out = gamma*x_norm+beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, gamma, beta, sample_mean, sample_var, x_norm, eps = cache
    N=x.shape[0]
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    dgamma=np.sum(dout * x_norm,axis=0)
    dbeta=np.sum(dout, axis=0)
    #就算是在代码中，这些东西也是一步一步来的。中间的每一个中间变量也要把ta的梯度记录下来
    dx_norm=dout * gamma
    # Denom = sqrt(sample_var + eps) 中间的标准差
    denom = np.sqrt(sample_var + eps)

    # Gradients w.r.t. x_centered and sample_var from normalization step
    dx_centered = dx_norm / denom  # ∂x_norm/∂x_centered = 1/denom
    dsample_var = np.sum(dx_norm * (x - sample_mean), axis=0) * (-0.5 / (denom ** 3))  # Chain: ∂x_norm/∂var = -0.5 * (var+eps)^{-1.5} * x_centered, summed
    
    # Additional gradient to x_centered from sample_var dependency
    dx_centered += 2 * (x - sample_mean) * dsample_var / N  # ∂var/∂x_centered = 2 * x_centered / N, broadcasted
    
    # Gradient w.r.t. sample_mean from centering
    dsample_mean = -np.sum(dx_centered, axis=0)
    
    # Full dx: from centering + from mean (sum branches)
    dx = dx_centered + dsample_mean / N  # Broadcast dsample_mean across N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # 感觉本质上只是将中间的量给放到一个公式里面而不是单独计算了
    x, gamma, beta, sample_mean, sample_var, x_norm, eps = cache
    N = x.shape[0]
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    dx_norm = dout * gamma
    ivar = 1.0 / np.sqrt(sample_var + eps)
    dx = ivar * (dx_norm - np.mean(dx_norm, axis=0) - x_norm * np.mean(dx_norm * x_norm, axis=0)) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # 不需要进行求矩阵转置的操作，否则gamma*x_norm的维度不匹配
    sample_mean = np.mean(x, axis=1, keepdims=True)
    # 求var和在BN里面的不一样
    sample_var = np.var(x, axis=1, keepdims=True)
    x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * x_norm + beta # x_norm还是N,D 没有问题
    cache = (x, gamma, beta, sample_mean, sample_var, x_norm, eps)  # Cache for backward
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    x, gamma, beta, sample_mean, sample_var, x_norm, eps = cache
    N, D = x.shape
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_norm = dout * gamma
    denom = np.sqrt(sample_var + eps)
    dx_centered = dx_norm / denom
    dsample_var = np.sum(dx_norm * (x - sample_mean), axis=1, keepdims=True) * (-0.5 / (denom ** 3))
    dx_centered += 2 * (x - sample_mean) * dsample_var / D
    dsample_mean = np.sum(dx_centered * (-1.0 / D), axis=1, keepdims=True)
    
    # Final dx
    dx = dx_centered + dsample_mean
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask=(np.random.rand(*x.shape)<p)/p
        out=x*mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out=x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx=dout*mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    # 每一个卷积核处理之后的输出的尺寸
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    out = np.zeros((N, F, H_out, W_out))
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    for n in range(N):
        # 循环遍历每一个滤波器（卷积核）
        for f in range(F):
            # 循环遍历输出特征图的每一个位置（高）
            for i in range(H_out):
                # 循环遍历输出特征图的每一个位置（宽）
                for j in range(W_out):
                    # (i, j) 是输出的位置, 我们需要找到它在输入(填充后)上的对应区域
                    
                    # 计算输入区域的起始位置
                    h_start = i * stride
                    w_start = j * stride
                    
                    # 从填充后的输入中切片出当前卷积核覆盖的区域
                    # 这个区域的维度是 (C, HH, WW)
                    input_slice = x_padded[n, :, h_start : h_start + HH, w_start : w_start + WW]
                    
                    # 获取当前的卷积核，维度是 (C, HH, WW)
                    current_filter = w[f, :, :, :]
                    
                    # 执行核心操作：对应元素相乘，然后全部求和
                    # 不同通道的加在一起了
                    conv_sum = np.sum(input_slice * current_filter)
                    
                    # 加上偏置项
                    result = conv_sum + b[f]
                    
                    # 将结果存入输出矩阵的对应位置
                    out[n, f, i, j] = result
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    x, w, b, conv_param = cache
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # 1. 准备工作：获取各种尺寸和参数
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # 获取输出尺寸
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # 2. 初始化梯度为零矩阵
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # 3. 对输入x进行补零，这对于计算dx至关重要
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    # 我们也需要一个带padding的dx版本来累加梯度
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)

    # 4. 开始循环计算梯度
    # 遍历每一个输入样本
    for n in range(N):
        # 遍历每一个滤波器
        for f in range(F):
            # 遍历输出的每一个位置
            for i in range(H_out):
                for j in range(W_out):
                    # 定位到当前计算所对应的输入切片
                    h_start = i * stride
                    w_start = j * stride
                    input_slice = x_padded[n, :, h_start : h_start + HH, w_start : w_start + WW]
                    
                    # 获取当前位置的上游梯度
                    current_dout = dout[n, f, i, j]

                    # ----- 计算梯度 -----
                    
                    # 计算db: 将所有输出点的梯度累加到对应的偏置上
                    db[f] += current_dout
                    
                    # 计算dw: 输入切片 * 上游梯度，累加到对应的滤波器梯度上
                    dw[f, :, :, :] += input_slice * current_dout
                    
                    # 计算dx: 滤波器 * 上游梯度，累加到对应的输入区域梯度上
                    # 注意：是累加到dx_padded上
                    dx_padded[n, :, h_start : h_start + HH, w_start : w_start + WW] += w[f, :, :, :] * current_dout

    # 5. 计算完所有梯度后，dx需要“裁剪”掉之前为了方便计算而填充的部分
    dx = dx_padded[:, :, pad:-pad, pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    N, C, H, W = x.shape
    stride = pool_param["stride"]
    pool_height=pool_param["pool_height"]
    pool_width=pool_param["pool_width"]
    H_out=1+(H-pool_height)//stride
    W_out=1+(W-pool_width)//stride
    out = np.zeros((N, C, H_out, W_out))
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    for n in range(N):
      for c in range(C):
        for i in range(H_out):
          # 遍历输出矩阵的每一个位置 (列)
          for j in range(W_out):
            # 1. 确定当前池化窗口在输入x上的范围
            h_start = i * stride
            h_end = h_start + pool_height
            w_start = j * stride
            w_end = w_start + pool_width
            
            # 2. 从输入x中切片出这个区域
            pool_region = x[n, c, h_start:h_end, w_start:w_end]
            
            # 3. 计算该区域的最大值
            max_value = np.max(pool_region)
            
            # 4. 将最大值存入输出矩阵的对应位置
            out[n, c, i, j] = max_value
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param=cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    # 获取输出尺寸
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    
    # 2. 初始化输入梯度dx为全零
    dx = np.zeros_like(x)
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    for n in range(N):
        for c in range(C):
            # 遍历输出的每一个位置（即dout的每一个元素）
            for i in range(H_out):
                for j in range(W_out):
                    # 确定当前池化窗口在输入x上的范围
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width
                    
                    # 从原始输入x中切片出这个区域
                    pool_region = x[n, c, h_start:h_end, w_start:w_end]
                    
                    # 找到该区域的最大值
                    max_val = np.max(pool_region)
                    
                    # 创建一个布尔掩码(mask)，标记出最大值所在的位置
                    # (pool_region == max_val) 会生成一个布尔矩阵，最大值位置为True，其余为False
                    mask = (pool_region == max_val)
                    
                    # 获取当前位置的上游梯度
                    grad = dout[n, c, i, j]
                    
                    # 将上游梯度应用到dx中对应的区域
                    # 只有mask中为True的位置（即原最大值位置）才会接收到梯度
                    dx[n, c, h_start:h_end, w_start:w_end] += mask * grad
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # 获取维度
    N, C, H, W = x.shape
    
    # 1. 将输入从 (N, C, H, W) 变形为 (N*H*W, C) 以便送入标准BN函数
    # 首先转置，让通道维度(C)在最后
    x_transposed = x.transpose(0, 2, 3, 1)
    # 然后变形
    x_reshaped = x_transposed.reshape(N * H * W, C)
    
    # 2. 调用标准的批量归一化函数
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    
    # 3. 将输出变形回原始的 (N, C, H, W) 形状
    # 首先变形回 (N, H, W, C)
    out_transposed = out_reshaped.reshape(N, H, W, C)
    # 然后转置回 (N, C, H, W)
    out = out_transposed.transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # 获取维度
    N, C, H, W = dout.shape
    
    # 1. 将上游梯度 dout 从 (N, C, H, W) 变形为 (N*H*W, C)
    # 步骤与前向传播完全相同
    dout_transposed = dout.transpose(0, 2, 3, 1)
    dout_reshaped = dout_transposed.reshape(N * H * W, C)
    
    # 2. 调用标准的批量归一化反向传播函数
    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    
    # 3. 将计算出的输入梯度 dx 变形回原始的 (N, C, H, W) 形状
    dx_transposed = dx_reshaped.reshape(N, H, W, C)
    dx = dx_transposed.transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    N, C, H, W = x.shape
    
    # 1. 将输入变形，把通道分成G组
    # (N, C, H, W) -> (N, G, C/G, H, W)
    x_reshaped = x.reshape(N, G, C // G, H, W)
    
    # 2. 沿着组内维度(C/G, H, W)计算均值和方差
    # keepdims=True 方便后续广播操作
    mean = np.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_reshaped, axis=(2, 3, 4), keepdims=True)
    
    # 3. 进行归一化
    x_normalized = (x_reshaped - mean) / np.sqrt(var + eps)
    
    # 4. 将形状还原为 (N, C, H, W)
    x_normalized_reshaped = x_normalized.reshape(N, C, H, W)
    
    # 5. 应用缩放(gamma)和平移(beta)
    out = gamma * x_normalized_reshaped + beta
    
    # 6. 存储反向传播需要用到的中间变量
    cache = (x, x_reshaped, x_normalized, G, mean, var, eps, gamma)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    x, x_reshaped, x_normalized, G, mean, var, eps, gamma = cache
    N, C, H, W = x.shape
    
    # 1. 计算dgamma和dbeta
    # gamma和beta的形状是 (1, C, 1, 1)，所以需要变形后的x_normalized
    x_normalized_reshaped = x_normalized.reshape(N, C, H, W)
    dgamma = np.sum(dout * x_normalized_reshaped, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    
    # 2. 计算反向传播到x_normalized的梯度
    dx_normalized = dout * gamma
    
    # 3. 将梯度变形回分组的形状 (N, G, C/G, H, W)
    dx_normalized_reshaped = dx_normalized.reshape(N, G, C // G, H, W)
    
    # 4. 反向传播通过归一化操作（这部分是标准归一化反传的链式法则）
    # M是每个组内的元素数量
    M = (C // G) * H * W
    
    # 计算dvar
    dvar = np.sum(dx_normalized_reshaped * (x_reshaped - mean) * -0.5 * (var + eps)**(-1.5), axis=(2, 3, 4), keepdims=True)
    
    # 计算dmean
    dmean_term1 = np.sum(dx_normalized_reshaped * (-1 / np.sqrt(var + eps)), axis=(2, 3, 4), keepdims=True)
    dmean_term2 = dvar * np.sum(-2 * (x_reshaped - mean), axis=(2, 3, 4), keepdims=True) / M
    dmean = dmean_term1 + dmean_term2
    
    # 计算dx_reshaped (输入x的梯度，但还是分组形状)
    dx_reshaped = (dx_normalized_reshaped / np.sqrt(var + eps)) + \
                  (dvar * 2 * (x_reshaped - mean) / M) + \
                  (dmean / M)
                  
    # 5. 将梯度dx_reshaped还原为 (N, C, H, W)
    dx = dx_reshaped.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
