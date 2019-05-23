import numpy as np


def sigmoid(X, deriv=False):
    if(deriv == True):
        return X*(1-X)
    return 1/(1 + np.exp(-X))


# 构造数据集
x = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1],
              [0, 0, 1]])

y = np.array([[0], [1], [1], [0], [0]])

# 指定随机的种子,每次运行产生相同数据
np.random.seed(1)
# L0有三个特征，L1有4个神经元，所以w0为3行4列,取值范围（-1，1）
w0 = 2*np.random.random((3, 4))-1
w1 = 2*np.random.random((4, 1))-1
# print(w0)

if __name__ == '__main__':
    # 神经网络模型构造及训练
    for j in range(1000001):
        # L0层
        l0 = x
        # 前向传播，计算后l1为5行4列
        l1 = sigmoid(np.dot(l0, w0))
        # 前向传播，计算后l2为5行1列
        l2 = sigmoid(np.dot(l1, w1))
        # 计算预测值与标签的差异值
        l2_error = y - l2
        if(j % 100000) == 0:
            print('Error'+str(np.mean(np.abs(l2_error))))
            print(j)

        # 反向传播，计算梯度值
        # 如l2_error很大，则需要大力度更新；如果l2_error很小，则只需要更新一点点
        # 所以导数乘以l2_error, *为连个5行一列的矩阵对应位置相乘
        l2_delta = l2_error * sigmoid(l2, deriv=True)
        # l2_delta5行1列，w14行1列，
        l1_error = l2_delta.dot(w1.T)
        l1_delta = l1_error * sigmoid(l1, deriv=True)

        # 更新参数
        # l1 5行4列，l2_delta5行1列，w1为4行一列
        w1 += l1.T.dot(l2_delta)
        # l0 5行3列，l2_delta5行4列，w0为3行4列
        w0 += l0.T.dot(l1_delta)
    print("w0=", w0)
    print("w1=", w1)
