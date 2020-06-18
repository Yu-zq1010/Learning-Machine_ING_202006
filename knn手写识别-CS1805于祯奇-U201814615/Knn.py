# CS1805班 于祯奇 U201814615

import operator
import struct
import numpy as np

# 训练集
train_images_file = "C:/Users/Yu-zq/Desktop/于祯奇/学习/大二/机器学习/knn-手写识别 CS1805于祯奇-U201814615/" \
                    "train-images.idx3-ubyte"
# 训练集标签
train_labels_file = "C:/Users/Yu-zq/Desktop/于祯奇/学习/大二/机器学习/knn-手写识别 CS1805于祯奇-U201814615/" \
                    "train-labels.idx1-ubyte"
# 测试集
test_images_file = "C:/Users/Yu-zq/Desktop/于祯奇/学习/大二/机器学习/knn-手写识别 CS1805于祯奇-U201814615/" \
                   "t10k-images.idx3-ubyte"
# 测试集标签
test_labels_file = "C:/Users/Yu-zq/Desktop/于祯奇/学习/大二/机器学习/knn-手写识别 CS1805于祯奇-U201814615/" \
                   "t10k-labels.idx1-ubyte"


def decode_images(idx3_ubyte_file):
    """解析图片"""
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0  # 偏移量初始为0，即从头开始读
    fmt_header = '>iiii'  # 按照大顶端方式读入四个int型变量
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # 读入的前四个变量分别为：魔数、图片数量、行数、列数

    # 解析数据集
    image_size = num_rows * num_cols  # 图片大小
    offset += struct.calcsize(fmt_header)  # 计算当前偏移量，即四个int类型的大小
    fmt_image = '>' + str(image_size) + 'B'  # 接着读入image_size个unsigned char类型的数，即每个数只有一个字节
    images = np.empty((num_images, num_rows, num_cols))  # 创建一个空的三维向量，每个维度分别代表着图片序号、行序号和列序号
    # 接下来是将读入的数据存放在刚刚创建的image数组中
    print("\n\n")
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))  # 读入
        for j in range(num_rows):  # 将读入的数据二值化
            for k in range(num_cols):
                if (images[i][j][k] > 0):
                    images[i][j][k] = 1
        offset += struct.calcsize(fmt_image)  # 更改偏移地址
    return images


def decode_lables(idx1_ubyte_file):
    """解析标签"""
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0  # 偏移量初始为0，即从头开始读
    fmt_header = '>ii'  # 按照大顶端方式读入两个int型变量
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # 读入的前两个变量分别为：魔数、图片数量

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'  # 接着读入一个unsigned char类型的数，即每个数只有一个字节
    labels = np.empty(num_images)  # 创建一个空的标签数组
    for i in range(num_images):  # 提取每个图片的标签
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]  # 由于是返回的元组，[0]是将其数据提取出来
        offset += struct.calcsize(fmt_image)  # 更改偏移地址
    return labels


def classify(inX, dataset, labels, k):
    """计算距离并排序"""
    datasetsize = dataset.shape[0]  # 读取矩阵第一维的长度，即训练集的样本数量
    ###以下距离计算公式
    diffMat = np.tile(inX, (datasetsize, 1)) - dataset  # 首先要将待测试的向量利用tile函数扩展成与训练集维度相同的向量，并做差
    sqDiffMat = diffMat ** 2  # 每一个相应维度的差做平方
    sqDistances = sqDiffMat.sum(axis=1)  # 将向量的第一维的每个数进行相加
    distances = sqDistances ** 0.5  # 最后将结果进行开方
    ###以上是距离计算公式

    # 距离从大到小排序，并返回其索引
    sortedDistIndicies = distances.argsort()
    # 字典
    classCount = {}
    # 前K个距离最小的
    for i in range(k):
        # sortedDistIndicies[0]返回的是距离最小的数据样本的序号
        # labels[sortedDistIndicies[0]]距离最小的数据样本的标签
        voteIlabel = labels[sortedDistIndicies[i]]
        # 以标签为key,支持该标签+1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 统计该标签出现的次数
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 第一个参数返回的是字典中的键值对；第二个参数是一个函数，其作用是查找作用于该函数的元组的第二个值；第三个参数代表降序排列
    return sortedClassCount[0][0]  # 返回的是最大数的标签，所以是[0][0]


if __name__ == '__main__':
    print("正在处理文件，请稍候……")
    train_images = decode_images(train_images_file)  # 载入训练集图片
    train_labels = decode_lables(train_labels_file)  # 载入训练集标签
    test_images = decode_images(test_images_file)  # 载入测试集图片
    test_labels = decode_lables(test_labels_file)  # 载入测试集标签
    status = True  # 状态变量，表示当前测试是否结束
    TestResult = {}  # 存储测试结果的字典
    num = 0  # 当前测试次数
    while (status):
        num += 1
        TrainNum = int(input("请输入训练集个数，最大为60000张："))
        K = int(input("请输入希望取得k值："))
        TestNum = int(input("请输入测试集个数，最大为10000张："))

        # 创建一个读入数据的数组，进行图片信息的记录
        trainingMat = np.zeros((TrainNum, 28 * 28))  # 置为零

        for i in range(TrainNum):  # 将原来返回的三维向量的后两维合并，变成一维
            for j in range(28):
                for k in range(28):
                    trainingMat[i, 28 * j + k] = train_images[i][j][k]

        errorCount = 0.0
        for i in range(TestNum):  # 进行测试
            classNumStr = test_labels[i]
            vectorUnderTest = np.zeros(784)
            for j in range(28):
                for k in range(28):
                    vectorUnderTest[28 * j + k] = test_images[i][j][k]  # 第i幅测试图

            Result = classify(vectorUnderTest, trainingMat, train_labels, K)
            print("正在识别第" + str(i) + "个，识别结果：%d 正确答案：%d" % (Result, classNumStr))
            if (Result != classNumStr):
                errorCount += 1.0
                print("错误")
        TestResult[str(num - 1)] = [TrainNum, TestNum, K, errorCount]
        Temp = input("若希望退出此次识别，请输入quit，输入其他则将继续：")
        if Temp == "quit":
            status = False

    for i in range(num):
        print("\n第" + str(i + 1) + "次训练集数量为：" + str(TestResult[str(i)][0]) + " 个，测试集数量为："
              + str(TestResult[str(i)][1]) + " 个，k值为：" + str(TestResult[str(i)][2]))
        print("错误数量为： %d" % TestResult[str(i)][3])
        print("正确率为： %.3f" % (1 - (TestResult[str(i)][3] / float(TestResult[str(i)][1]))))
    print("\n\n手写识别结束")
