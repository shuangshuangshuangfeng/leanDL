import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, nn
import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

"""
AlexNet, 2012年， 使用了8层卷积神经网络，首次证明了学习到的特征可以超越手工设计的特征，从而一举打破计算机视觉研究的前状。
AlexNet在LeNet的基础上加了三个卷积层，但是AlexNet作者对于他们的卷积窗口，输出通道和构造顺序做了大量的调整
虽然它知名了深度卷积神经网络可以取得出色的结果，但是并没有提供简单的规则以指导后来的研究者如何设计新的网络。
"""


"""
网络构建
"""
net = nn.Sequential()
# 使用较大的11 x 11窗口来捕获物体。同时使用步幅4来较大幅度减小输出高和宽。这里使用的输出通
# 道数比LeNet中的也要大很多
# 第一阶段
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'))
net.add(nn.MaxPool2D(pool_size=3, strides=2))

# 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
# 第二阶段
net.add(nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'))
net.add(nn.MaxPool2D(pool_size=3, strides=2))
# 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
# 前两个卷积层后不使用池化层来减小输入的高和宽


# 第三阶段
net.add(nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'))
net.add(nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'))
net.add(nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'))
net.add(nn.MaxPool2D(pool_size=3, strides=2))

# 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
# 第四阶段
net.add(nn.Flatten())
net.add(nn.Dense(4096, activation="relu"))
net.add(nn.Dropout(0.5))

# 第五阶段
net.add(nn.Dense(4096, activation="relu"))
net.add(nn.Dropout(0.5))

# 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
# 第六阶段
net.add(nn.Dense(10))


"""
构造一个高和宽均为224的单通道数据样本来观察每一层的输出形状
"""
X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)


"""
读取数据集
"""
# 本函数已保存在d2lzh包中方便以后使用
def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.mxnet', 'datasets', 'fashion-mnist')):
    root = os.path.expanduser(root)  # 展开用户路径'~'
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(
        mnist_train.transform_first(transformer), batch_size, shuffle=True,
        num_workers=num_workers)
    test_iter = gdata.DataLoader(
        mnist_test.transform_first(transformer), batch_size, shuffle=False,
        num_workers=num_workers)
    return train_iter, test_iter

batch_size = 1
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = load_data_fashion_mnist(batch_size)


"""
训练
"""
lr, num_epochs, ctx = 0.01, 5, d2l.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)







