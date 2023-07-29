import numpy as np
import time

## TENSORS

from tinygrad.tensor import Tensor

t1 = Tensor([1, 2, 3, 4, 5])
na = np.array([1, 2, 3, 4, 5])
t2 = Tensor(na)

full = Tensor.full(shape=(2, 3), fill_value=5)
zeros = Tensor.zeros(2, 3)
ones = Tensor.ones(2, 3)

full_like = Tensor.full_like(full, fill_value=2)
zeros_like = Tensor.zeros_like(full)
ones_like = Tensor.ones_like(full)

eye = Tensor.eye(3) # 3x3 identity matrix
arrange = Tensor.arange(start=0, stop=10, step=1)

rand = Tensor.rand(2, 3)
randn = Tensor.randn(2, 3)
uniform = Tensor.uniform(2, 3, low=0, high=10)

from tinygrad.helpers import dtypes
t3 = Tensor([1, 2, 3, 4, 5], dtype=dtypes.int32)


t5 = (t1 + 1) * 2
t6 = (t1 * t5)
print (t6.numpy())
t6 = t6.relu()
print (t6.numpy())
t6 = t6.log_softmax()
print (t6.numpy())


## MODELS

class Linear:
    def __init__(self, in_features, out_features, bias=True, initialization: str='kaiming_uniform'):
        self.weight = getattr(Tensor, initialization)(out_features, in_features)
        self.bias = Tensor.zeros(out_features) if bias else None

    def __call__(self, x):
        return x.linear(self.weight.transpose(), self.bias)


class TinyNet:
    def __init__(self):
        self.l1 = Linear(784, 128, bias=False)
        self.l2 = Linear(128, 10, bias=False)

    def __call__(self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        return x.log_softmax()

net = TinyNet()

Tensor.training = True

def cross_entropy(out, Y):
    num_classes = out.shape[-1]
    YY = Y.flatten().astype(np.int32)
    y = np.zeros((YY.shape[0], num_classes), np.float32)
    y[range(y.shape[0]),YY] = -1.0*num_classes
    y = y.reshape(list(Y.shape)+[num_classes])
    y = Tensor(y)
    return out.mul(y).mean()
