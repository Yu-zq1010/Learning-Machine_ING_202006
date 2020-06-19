# encoding=utf-8
import re
import pickle
import jieba;
import os;


class spamEmailBayes:
    def getStopWords(self):
        stopList = []
        for line in open("../数据集/中英文停用词表", encoding='UTF-8'):  # 获得停用词表
            stopList.append(line[:len(line) - 1])
        return stopList;

    def get_word_list(self, content, wordsList, stopList):
        res_list = list(jieba.cut(content))  # 分词结果放入res_list
        for i in res_list:
            if i not in stopList and i.strip() != '' and i != None:
                if i not in wordsList:
                    wordsList.append(i)

    def addToDict(self, wordsList, wordsDict):  # 生成词典
        for item in wordsList:
            if item in wordsDict.keys():  # 若列表中的词已在词典中，则加1，
                wordsDict[item] += 1
            else:  # 否则添加进去
                wordsDict.setdefault(item, 1)

    def get_File_List(self, filePath):
        filenames = os.listdir(filePath)
        return filenames

    def getTestWords(self, testDict, spamDict, normDict, normFilelen, spamFilelen):  # 计算贝叶斯概率（p（s|w））
        spamdictlen = len(spamDict)  # 垃圾邮件词类数目
        normdictlen = len(normDict)  # 正常邮寄词类数目
        wordProbList = {}

        for word, num in testDict.items():
            if word in spamDict.keys() and word in normDict.keys():  # 垃圾和正常都有这个词
                pw_s = spamDict[word] / spamFilelen
                pw_n = normDict[word] / normFilelen
                ps_w = pw_s / (pw_s + pw_n)
                wordProbList.setdefault(word, ps_w)
            if word in spamDict.keys() and word not in normDict.keys():  # 垃圾
                pw_s = spamDict[word] / spamFilelen
                pw_n = 1 / (normFilelen + normdictlen)  # 拉普拉斯平滑
                ps_w = pw_s / (pw_s + pw_n)
                wordProbList.setdefault(word, ps_w)
            if word not in spamDict.keys() and word in normDict.keys():  # 正常
                pw_s = 1 / (spamFilelen + spamdictlen)  # 拉普拉斯平滑
                pw_n = normDict[word] / normFilelen
                ps_w = pw_s / (pw_s + pw_n)
                wordProbList.setdefault(word, ps_w)
            if word not in spamDict.keys() and word not in normDict.keys():  # 不在词典中
                pw_s = 1 / (spamFilelen + spamdictlen)  # 拉普拉斯平滑
                pw_n = 1 / (normFilelen + normdictlen)
                ps_w = pw_s / (pw_s + pw_n)
                wordProbList.setdefault(word, ps_w)
        sorted(wordProbList.items(), key=lambda d: d[1], reverse=True)[0:15]
        return wordProbList

    def calBayes(self, wordList, spamdict, normdict):  # 计算联合概率
        ps_w = 1
        ps_n = 1
        for word, prob in wordList.items():
            ps_w *= prob
            ps_n *= (1 - prob)
        if ps_w + ps_n == 0:
            return 0.95
        p = ps_w / (ps_w + ps_n)
        return p

    def calAccuracy(self, testResult):  # 计算预测结果正确率
        rightCount = 0
        errorCount = 0
        for name, catagory in testResult.items():
            if (int(name) < 100000 and catagory == 0) or (int(name) > 100000 and catagory == 1):
                rightCount += 1
            else:
                errorCount += 1
        return rightCount / (rightCount + errorCount)


spam = spamEmailBayes()
wordsList = []  # 词语的集合（每个词只出现一次）
wordsDict = {}  # 词语的集合（每个词只出现一次）
testResult = {}  # 保存预测结果,key为文件名，值为预测类别

spamDict = {}  # 垃圾邮件词类及每个词的数量
normDict = {}  # 正常邮件词类及每个词的数量
testDict = {}  # 测试邮件词类及每个词的数量

normFileList = spam.get_File_List("../数据集/非垃圾邮件")  # 正常邮件的列表
spamFileList = spam.get_File_List("../数据集/垃圾邮件")  # 垃圾邮件的列表
testFileList = spam.get_File_List("../测试集")  # 测试邮件的列表

normFilelen = len(normFileList)  # 正常邮件数量
spamFilelen = len(spamFileList)  # 垃圾邮件的数量

stopList = spam.getStopWords()  # 停用词过滤

print("正在解析19338个非垃圾邮件，请稍候……")
i = 0
for fileName, i in zip(normFileList, range(len(normFileList))):  # 获得正常邮件中的词频
    if (i + 1) % 1000 == 0:
        print("已读取%d个文件" % (i + 1))
    wordsList.clear()
    for line in open("../数据集/非垃圾邮件/" + fileName, encoding='gbk', errors='ignore'):  # 更改1
        rule = re.compile(r"[^\u4e00-\u9fa5]")  # 过滤掉非中文字符
        line = rule.sub("", line)
        spam.get_word_list(line, wordsList, stopList)  # 将每封邮件出现的词保存在wordsList中
    spam.addToDict(wordsList, wordsDict)  # 统计每个词在所有邮件中出现的次数
normDict = wordsDict.copy()
output = open('非垃圾邮件.pkl', 'wb')
pickle.dump(normDict, output, -1)
output.close()
print("非垃圾邮件解析结束")

print("正在解析37661个垃圾邮件，请稍候……")
i = 0
wordsDict.clear()
for fileName, i in zip(spamFileList, range(len(spamFileList))):  # 获得垃圾邮件中的词频
    if (i + 1) % 1000 == 0:
        print("已读取%d个文件" % (i + 1))
    wordsList.clear()
    for line in open("../数据集/垃圾邮件/" + fileName, encoding='gbk', errors='ignore'):  # genggai2
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        line = rule.sub("", line)
        spam.get_word_list(line, wordsList, stopList)
    spam.addToDict(wordsList, wordsDict)
spamDict = wordsDict.copy()
output = open('垃圾邮件.pkl', 'wb')
pickle.dump(spamDict, output, -1)
output.close()
output = open('模型.pkl', 'wb')
pickle.dump(spam, output, -1)
output.close()

Testnum = input("请输入测试集个数：")
i = 0
for fileName, i in zip(testFileList, range(int(Testnum))):  # 分类测试邮件
    testDict.clear()
    wordsDict.clear()
    wordsList.clear()
    for line in open("../测试集/" + fileName, encoding='gbk', errors='ignore'):  # genggai3
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        line = rule.sub("", line)
        spam.get_word_list(line, wordsList, stopList)
    spam.addToDict(wordsList, wordsDict)
    testDict = wordsDict.copy()
    wordProbList = spam.getTestWords(testDict, spamDict, normDict, normFilelen, spamFilelen)  # 计算贝叶斯概率（p（s|w））
    p = spam.calBayes(wordProbList, spamDict, normDict)  # 计算联合概率
    if (p > 0.9):  # 设定阈值
        testResult.setdefault(fileName, 1)
    else:
        testResult.setdefault(fileName, 0)

testAccuracy = spam.calAccuracy(testResult)  # 计算分类准确率（测试集中文件名低于7000的为正常邮件）
for i, ic in testResult.items():
    if ((str(ic)) == '0'):
        print('项目名称:', i, "——结果：正常邮件");
        if i > 100000:
            print("分类错误")
    else:
        print('项目名称:', i, "——结果：垃圾邮件");
        if i < 100000:
            print("分类错误")
print("准确率：%2.2f%%" % (100.0 * testAccuracy))
