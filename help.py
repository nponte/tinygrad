import numpy as np
import time

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
