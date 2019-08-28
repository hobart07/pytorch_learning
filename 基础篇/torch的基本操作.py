import torch
import tensorflow as tf
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


def Dynamic_and_static():
    "--------tensorflow:while-----------"
    first_counter = tf.constant(0)
    second_counter = tf.constant(10)
    def cond(first_counter, second_counter, *args):
        return first_counter < second_counter
    def body(first_counter, second_counter):
        first_counter = tf.add(first_counter, 2)
        second_counter = tf.add(second_counter, 1)
        return first_counter, second_counter
    c1, c2 = tf.while_loop(cond, body, [first_counter, second_counter])
    with tf.Session() as sess:
        counter_1_res, counter_2_res = sess.run([c1, c2])
    print(counter_1_res)
    print(counter_2_res)
    "--------pytorch:while-----------"
    first_counter = torch.Tensor([0])
    second_counter = torch.Tensor([10])

    while (first_counter < second_counter)[0]:
        first_counter += 2
        second_counter += 1

    print(int(first_counter.numpy()[0]))
    print(second_counter)

def Automatic_derivative():
    # "--------demo:1-----------"
    # "简单函数求导：z=(x+2)**2+3  ∂z/∂x=2(x+2)=2(2+2)=8"
    # x = Variable(torch.Tensor([2]), requires_grad=True)
    # y = x + 2
    # z = y ** 2 + 3
    # print(z)
    # # 使用自动求导
    # z.backward()
    # print(x.grad)
    # "--------demo:2-----------"
    # x = Variable(torch.randn(10, 20), requires_grad=True)
    # y = Variable(torch.randn(10, 5), requires_grad=True)
    # w = Variable(torch.randn(20, 5), requires_grad=True)
    # out = torch.mean(y - torch.matmul(x, w)) # torch.matmul 是做矩阵乘法
    # out.backward()
    # # 得到 x 的梯度
    # print(x.grad)
    # # 得到 y 的的梯度
    # print(y.grad)
    # # 得到 w 的梯度
    # print(w.grad)
    # "--------demo:3-----------"
    # m = Variable(torch.FloatTensor([[2, 3]]), requires_grad=True) # 构建一个 1 x 2 的矩阵
    # n = Variable(torch.zeros(1, 2)) # 构建一个相同大小的 0 矩阵
    # print(m)
    # print(n)
    # # 通过 m 中的值计算新的 n 中的值
    # n[0, 0] = m[0, 0] ** 2
    # n[0, 1] = m[0, 1] ** 2
    # print(n)
    # n.backward(torch.ones_like(n)) # 将 (w0, w1) 取成 (1, 1)
    # print(m.grad)
    "--------demo:4-----------"
    x = Variable(torch.FloatTensor([2, 3]), requires_grad=True)
    k = Variable(torch.zeros(2))
    k[0] = x[0] ** 2 + 3 * x[1]
    k[1] = x[1] ** 2 + 2 * x[0]
    print(k)
    j = torch.zeros(2, 2)
    k.backward(torch.FloatTensor([1, 0]), retain_graph=True)
    j[0] = x.grad.data
    x.grad.data.zero_()  # 归零之前求得的梯度
    k.backward(torch.FloatTensor([0, 1]))
    j[1] = x.grad.data
    print(j)
def Variable_practice():
    "--------Variable-----------"
    x_tensor = torch.randn(4, 2)
    y_tensor = torch.randn(4, 2)
    x = Variable(x_tensor, requires_grad=True)  # 将 tensor 变成 Variable.默认 Variable 是不需要求梯度的，所以我们用这个方式申明需要对其进行求梯度
    y = Variable(y_tensor, requires_grad=True)
    z = torch.sum(x + y)
    print(z.data)    #Variable 中的 tensor本身值 `.data`
    print(z.grad_fn)    #Z是通过什么方式得到的 `.grad_fn`
    z.backward()
    print(x.grad)    #x tensor 的梯度`.grad`
    print(y.grad)    #y tensor 的梯度`.grad`
    "-----------demo----------"
    x = np.arange(-3, 3.01, 0.1)
    y = x ** 2
    plt.plot(x, y)
    plt.plot(2, 4, 'ro')
    plt.show()
    x = Variable(torch.FloatTensor([2]), requires_grad=True)
    y = x ** 2
    y.backward()
    print(x.grad)

def tensor_fly():
    "--------内部格式转换-----------"
    x = torch.ones(2, 2)
    print(x, x.dtype)  # 这是一个torch.float32
    x = x.long()
    print(x, x.dtype)  # 这是一个torch.int64
    x = x.float()
    print(x, x.dtype)  # 这是一个torch.int64
    "--------行列上的计算-----------"
    x = torch.randn(4, 3)
    print(x)
    # 沿着行取最大值
    max_value, max_idx = torch.max(x, dim=1)
    # 每一行的最大值
    print("max_value:", max_value)
    # 每一行最大值的下标
    print("max_idx:", max_idx)
    # 沿着行对 x 求和
    sum_x = torch.sum(x, dim=1)
    print("sum_x:", sum_x)
    "--------纬度的变化-----------"
    x = torch.randn(3, 4, 5)
    print(x.shape)
    x = x.unsqueeze(0)  # 在第一维增加
    print("在第一维增加:", x.shape)
    x = x.unsqueeze(1)  # 在第二维增加
    print("在第二维增加:", x.shape)
    x = x.squeeze(0)  # 减少第一维
    print("减少第一维:", x.shape)
    x = x.squeeze()  # 将 tensor 中所有的一维全部都去掉
    print("将 tensor 中所有的一维全部都去掉:", x.shape)
    x = x.permute(1, 0, 2)  # permute 可以重新排列 tensor 的维度
    print("permute 可以重新排列 tensor 的维度:", x.shape)
    x = x.transpose(0, 2)  # transpose 交换 tensor 中的两个维度
    print("transpose 交换 tensor 中的两个维度:", x.shape)
    "--------点乘与叉乘----------"
    x1 = torch.tensor([[1, 2], [1, 2]])
    y1 = x1 * x1
    z1 = torch.mul(x1, x1)  #   点乘  注：点乘求和即为卷积
    print(y1)
    print(z1)
    x2 = torch.tensor([[1, 2], [1, 2]])
    y2 = torch.mm(x2, x2)  #   叉乘
    print(y2)


def numpy_and_torch():
    "--------numpy转pytorch-----------"
    numpy_array = np.array(range(5))
    print(type(numpy_array), numpy_array.dtype, numpy_array)
    pytorch_tensor_1 = torch.Tensor(numpy_array)    #numpy格式装tensor格式（int型转为float型）
    print(type(pytorch_tensor_1), pytorch_tensor_1.dtype, pytorch_tensor_1)
    pytorch_tensor_2 = torch.from_numpy(numpy_array)    #numpy格式装tensor格式（int型不变）
    print(type(pytorch_tensor_2), pytorch_tensor_2.dtype, pytorch_tensor_2)
    "---------pytorch转numpy----------"
    # tensor转numpy变量类型不变
    numpy_array_1 = pytorch_tensor_1.numpy()
    numpy_array_2 = pytorch_tensor_2.numpy()
    print(numpy_array_1, numpy_array_1.dtype)
    print(numpy_array_2, numpy_array_2.dtype)
    #如果是在GPU上运行
    #numpy_array_gpu = pytorch_tensor_1.cpu.numpy()

def init_parameter(a = True, b = True):
    # torch的常数初始化
    x1 = torch.empty(2, 3)    #未初始化
    x2 = torch.zeros(2, 3)    #0初始化
    x3 = torch.ones(2, 3)    #1初始化
    x4 = torch.zeros_like(x1)
    x5 = torch.ones_like(x1)
    x6 = torch.arange(start = 0, end = 6, step=1)    #跟python的range()一样
    x7 = torch.full([8], 1)
    if a == True:
        print("未初始化:\n", x1, x1.dtype)
        print("0初始化:\n", x2, x2.dtype)
        print("1初始化:\n", x3, x3.dtype)
        print("与输入唯独相同的0初始化:\n", x4, x4.dtype)
        print("与输入唯独相同的1初始化:\n", x5, x5.dtype)
        print("torch里的range:\n", x6, x6.dtype)
        print("torch里填充:\n", x7, x7.dtype)

    # torch的随机初始化
    x8 = torch.rand(2, 3)    #[0,1)内的均匀分布随机数
    x9 = torch.randn(2, 3)    #返回标准正太分布N(0,1)的随机数
    x10 = torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))  #均值为mean，标准差为std的正态分布
    if b == True:
        print("[0,1)内的均匀分布随机数:\n", x8, x8.dtype)
        print("返回标准正太分布N(0,1)的随机数:\n", x9, x9.dtype)
        print("均值为mean，标准差为std的正态分布:\n", x10, x10.dtype)

if __name__ == '__main__':
    # init_parameter()
    # tensor_fly()
    # numpy_and_torch()
    # Variable_practice()
    Automatic_derivative()
    # Dynamic_and_static()