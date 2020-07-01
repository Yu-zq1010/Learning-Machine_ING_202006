import pandas as pd
import numpy as np
import xlwt


# 更新参数，训练模型
def train(x_train, y_train, epoch, learning_rate, reg_rate):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    bias = 0  # 偏置值初始化
    weights = np.ones(dim)  # 权重初始化
    bg2_sum = 0  # 用于存放偏置值的梯度平方和
    wg2_sum = np.zeros(dim)  # 用于存放权重的梯度平方和
    Epsilon = 1e-5
    LossList = []

    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(dim)
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        for j in range(num):
            y_pre = weights.dot(x_train[j, :]) + bias
            sig = 1 / (1 + np.exp(-y_pre))
            b_g += (-1) * (y_train[j] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[j] - sig) * x_train[j, k] + 2 * reg_rate * weights[k]
        b_g /= num
        w_g /= num

        # adagrad
        bg2_sum += b_g ** 2
        wg2_sum += w_g ** 2
        # 更新权重和偏置
        bias -= learning_rate / bg2_sum ** 0.5 * b_g
        weights -= learning_rate / wg2_sum ** 0.5 * w_g

        # 在计算loss时，由于涉及到log()运算，因此可能出现无穷大，计算并打印出来的loss为nan

        loss = 0
        acc = 0
        result = np.zeros(num)
        for j in range(num):
            y_pre = weights.dot(x_train[j, :]) + bias
            sig = 1 / (1 + np.exp(-y_pre))
            if sig >= 0.5:
                result[j] = 1
            else:
                result[j] = 0

            if result[j] == y_train[j]:
                acc += 1.0
            loss += (-1) * (y_train[j] * np.log(sig + Epsilon) + (1 - y_train[j]) * np.log(1 - sig + Epsilon))
        LossList.append(loss / num)

    return weights, bias, LossList


# 验证模型效果
def validate(x_val, y_val, weights, bias):
    loss = 0
    acc = 0
    num = x_val.shape[0]
    result = np.zeros(num)

    for j in range(num):
        y_pre = weights.dot(x_val[j, :]) + bias
        sig = 1 / (1 + np.exp(-y_pre))
        if sig >= 0.5:
            result[j] = 1
        else:
            result[j] = 0


        if result[j] == y_val[j]:
            acc += 1.0
    return acc / num


def main():
    # 从csv中读取有用的信息
    df = pd.read_csv('数据集.csv')
    # 空值填0
    df = df.fillna(0)
    # (4000, 59)
    array = np.array(df)
    # (4000, 57)
    x = array[:, 1:-1]
    # scale
    x[:, -1] /= np.mean(x[:, -1])
    x[:, -2] /= np.mean(x[:, -2])
    # (4000, )
    y = array[:, -1]

    # 划分训练集与验证集
    x_train, x_val = x[:2999, :], x[2999:, :]
    y_train, y_val = y[0:2999], y[2999:]

    ExcelFile = xlwt.Workbook()

    # 1.不同学习率对于Loss的影响
    print("\t验证学习率对于Loss的影响：")
    epoch = 200
    sheet = ExcelFile.add_sheet('学习率对Loss的影响', cell_overwrite_ok=True)
    learning_rates = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
    reg_rate = 0.001
    # 准备写入工作
    sheet.write_merge(0, 1, 0, 0, '迭代次数')
    sheet.write_merge(0, 0, 1, len(learning_rates), 'Loss的值')
    for i in range(epoch):
        sheet.write(i + 2, 0, i + 1)

    for i in range(len(learning_rates)):
        print("正在计算学习率为%f时的Loss值" % learning_rates[i])
        sheet.write(1, i + 1, 'α = ' + str(learning_rates[i]))
        w, b, LossList = train(x_train, y_train, epoch, learning_rates[i], reg_rate)
        for j in range(epoch):
            sheet.write(j + 2, i + 1, LossList[j])

    # 2. 不同正则化参数对于Loss的影响
    print("\t验证正则化参数对于Loss的影响：")
    epoch = 50
    sheet = ExcelFile.add_sheet('正则化参数对Loss的影响', cell_overwrite_ok=True)
    learning_rate = 1
    reg_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    # 准备写入工作
    sheet.write_merge(0, 1, 0, 0, '迭代次数')
    sheet.write_merge(0, 0, 1, len(reg_rates), 'Loss的值')
    for i in range(epoch):
        sheet.write(i + 2, 0, i + 1)

    for i in range(len(reg_rates)):
        print("正在计算正则化参数为%f时的Loss值" % reg_rates[i])
        sheet.write(1, i + 1, 'r = ' + str(reg_rates[i]))
        w, b, LossList = train(x_train, y_train, epoch, learning_rate, reg_rates[i])
        for j in range(epoch):
            sheet.write(j + 2, i + 1, LossList[j])

    # 3.不同学习率对于正确率的影响
    print("\t验证学习率对于准确率的影响：")
    epoch = 100
    sheet = ExcelFile.add_sheet('学习率对准确率的影响', cell_overwrite_ok=True)
    learning_rates = np.linspace(0.01, 20, 200)
    reg_rate = 0.001
    # 准备写入工作
    sheet.write(0, 0, '学习率')
    sheet.write(0, 1, '准确率')

    for i in range(len(learning_rates)):
        print("正在计算学习率为%f时的准确率" % learning_rates[i])
        sheet.write(i + 1, 0, learning_rates[i])
        w, b, LossList = train(x_train, y_train, epoch, learning_rates[i], reg_rate)
        acc = validate(x_val, y_val, w, b)
        sheet.write(i + 1, 1, acc)

    # 4.不同正则化参数对于准确率的影响
    print("\t验证正则化参数对于准确率的影响：")
    epoch = 100
    sheet = ExcelFile.add_sheet('正则化参数对准确率的影响', cell_overwrite_ok=True)
    learning_rate = 1
    reg_rates = np.linspace(0, 0.01, 100)
    # 准备写入工作
    sheet.write(0, 0, '正则化参数')
    sheet.write(0, 1, '准确率')

    for i in range(len(reg_rates)):
        print("正在计算正则化参数为%f时的准确率" % reg_rates[i])
        sheet.write(i + 1, 0, reg_rates[i])
        w, b, LossList = train(x_train, y_train, epoch, learning_rate, reg_rates[i])
        acc = validate(x_val, y_val, w, b)
        sheet.write(i + 1, 1, acc)

    # 5.不同迭代次数对于准确率的影响
    print("\t验证迭代次数对于准确率的影响")
    epochs = np.linspace(10, 2000, 40, dtype=int)
    learning_rate = 1
    reg_rate = 0.001
    sheet = ExcelFile.add_sheet('迭代次数对于准确率的影响', cell_overwrite_ok=True)
    # 准备写入工作
    sheet.write(0, 0, '迭代次数')
    sheet.write(0, 1, '准确率')

    for i in range(len(epochs)):
        print("当前迭代次数为%d" % epochs[i])
        sheet.write(i + 1, 0, float(epochs[i]))
        w, b, LossList = train(x_train, y_train, epochs[i], learning_rate, reg_rate)
        acc = validate(x_val, y_val, w, b)
        sheet.write(i + 1, 1, acc)

    # 保存实验结果记录表
    ExcelFile.save('./实验结果.xls')


if __name__ == '__main__':
    main()
