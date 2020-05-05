import torch
from time import time

print(torch.__version__)
"""
# 线性回归
"""


a = torch.ones(1000)
b = torch.ones(1000)

"""
将两个向量按元素逐一做标量加法。
"""

start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)

"""
两个向量做矢量加法
"""
start = time()
d = a + b
print(time() - start)

"""
广播机制例子
"""
a = torch.ones(3)
b = 10
print(a + b)

"""
# 结论：
矢量是向量，含方向，标量只有大小没有方向，相当于一个数
标量加法是将向量中的每一个对应位置的数相加
矢量加法是将向量直接相加
"""




