import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random


print(torch.__version__)
"""
# 线性回归从零开始
"""



"""
1. 生成数据集
"""
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
# 返回一个0-1的序列，序列是从标准正态分布中随机取的
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
# 标签，也叫结果，是由特征做一定运算的来的
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# 运算的结果中稍微加一点噪声，也就是模糊一些的意思
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)

print(features[0], labels[0])

# def use_svg_display():
#     # 用矢量图显示
#     display.set_matplotlib_formats('svg')
#
# def set_figsize(figsize=(3.5, 2.5)):
#     use_svg_display()
#     # 设置图的尺寸
#     plt.rcParams['figure.figsize'] = figsize
#
# # # 在../d2lzh_pytorch里面添加上面两个函数后就可以这样导入
# # import sys
# # sys.path.append("..")
# # from d2lzh_pytorch import *
#
# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
"""
读取数据
"""
# 本函数已保存在d2lzh包中方便以后使用
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

"""
初始化模型参数
"""
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

# 自动求导
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

"""
定义模型
"""
def linreg(X, w, b):  # 本函数已保存在d2lzh包中方便以后使用
    # torch.mm(),两个向量相乘
    return torch.mm(X, w) + b

"""
定义损失函数
"""
def squared_loss(y_hat, y):  # 本函数已保存在pytorch_d2lzh包中方便以后使用
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

"""
定义优化算法
"""
def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

"""
训练模型
"""
lr = 0.03  # learning_rate 学习率
num_epochs = 7  # 对数据扫几遍
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))



