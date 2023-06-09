import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyecharts
import math
from sklearn.feature_selection import VarianceThreshold


def x_y_distance(x, y):
    return np.sqrt(np.sum(np.power(x - y, 2)))


# 随机生成k个质心
def random_k_centroid(dataset, k):
    data_center = []
    # 在原有列表的基础上选则k个簇心

    for i in range(k):
        data_append = list(dataset.iloc[np.random.randint(0, dataset.shape[0])])
        data_center.append(data_append)
    data_return = np.mat(data_center)
    return data_return


# 编写完整k-means聚类函数
# 第一个容器用于存放和更新质心，该容器可以考虑用list来执行，list不仅是可迭代的对象，同时list内不同元素索引的
# 位置也可用来标记和区分个质心，即各簇的编号
# 第二个容器用来记录，保存和更新各点到质心之间的距离，并能够方便对其进行比较考虑用数组来执行，第一列来存放
# 最近一次计算完成后某点到各质心的最短距离，第二列来记录计算完成后根据最短距离得到的代表对应质心的数值索引，质心的编号
# 第三列，用放上一次的编号，用于后续比较

def KMeans(dataset_dimension, k):
    m = dataset_dimension.shape[0]
    n = dataset_dimension.shape[1]  # 行和列
    centroid = random_k_centroid(dataset_dimension, k)
    # 创建两列，一列放类别，一列放距离

    clusterAssment = np.zeros((m, 2))
    clusterchang = True
    # 判断质心变化，某点是都归属另一个簇了
    while clusterchang:
        clusterchang = False
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            for m in range(k):
                way = x_y_distance(centroid[m, :], dataset_dimension.iloc[i, :].values)
                if way < min_dist:
                    min_dist = way
                    min_index = m
            if clusterAssment[i, 0] != min_index:
                clusterchang = True
                clusterAssment[i, :] = min_index, min_dist ** 3
        for j in range(k):
            put_cluster = dataset_dimension[clusterAssment[:, 0] == j]  # 将属于簇心的j拿出来
            centroid[j, :] = np.mean(put_cluster, axis=0)
    return centroid, clusterAssment


def print_means(dataset, k):
    centroids, clusterassment = KMeans(dataset, k)
    dataset['label'] = clusterassment[:, 0]
    color = ['m', 'g', 'b', 'c', 'y']
    for i in range(k):
        x = list()
        y = list()
        for j in range(dataset.shape[0]):
            if (dataset.iloc[j, dataset.shape[1] - 1] == i):
                x.append(dataset.iloc[j, 0])
                y.append(dataset.iloc[j, 1])
        plt.scatter(x, y, c=color[i])
    plt.scatter(list(centroids[:, 0]), list(centroids[:, 1]), color='red', marker='x', s=200)
    plt.show()


# def FM(newlabel, oldlabel):
#     a = b = c = d = 0
#     for i in range(0, newlabel.shape[0]):
#         for j in range(i + 1, newlabel.shape[0]):
#             if newlabel[i] == newlabel[j] and oldlabel[i] == oldlabel[j]:
#                 a += 1
#             elif newlabel[i] == newlabel[j] and oldlabel[i] != oldlabel[j]:
#                 b += 1
#             elif newlabel[i] != newlabel[j] and oldlabel[i] == oldlabel[j]:
#                 c += 1
#             elif newlabel[i] != newlabel[j] and oldlabel[i] != oldlabel[j]:
#                 d += 1
#     print("聚类结果的评估指数为(FM指数)：", math.sqrt((a / (a + b)) * (a / (a + c))))


# 误差平方和sse
def sse(dataset_dimension):
    sse = list()
    for i in range(2, 9):
        centroid, result_set = KMeans(dataset_dimension, i)
        sse.append(np.sum(result_set[:, 1]))
        print('k = ' + str(i) + " sse = " + str(sum(result_set[:, 1])))

    plt.plot(range(2, 9), sse, '--o')
    plt.show()


模型收敛稳定性
def stability(dataset,k):
  np.random.seed(123)
  for i in range(1,5):
     plt.subplot(2,2,i)
     centroid,cluster = KMeans(dataset,k)
     plt.scatter(cluster.iloc[:,0],cluster.iloc[:,1], c = cluster.iloc[:,-1],)
     plt.plot(centroid[:,0],centroid[:,1],'o',color = 'red')
     print(cluster.iloc[:,3].sum())
# 初始质心的随机选取会影响聚类的最终结果
# 质心数量在某种程度上影响随机的程度，如果质心数量选取和数据空间分布相似，那么对最终聚类的结果影响比较小


# 解决方案，在设置质心随机生成的过程中，尽可能使质心更加分散，或者手动初始质心，降低随机性影响，挥着增量更新质心
# 在点到簇的每次指派后，增量地更新质心，而不是在所有的点都指派到簇之后再更新，每步需要零次或者两次质心的更新，移到别的簇或者留到当前簇
# 使用增量更新确保不会产生空簇，所有簇都从单点开始，并且如果一个簇只有一个单点，那么总是被放到相同的簇
# 此外，如果使用增量更新，可以调整点的相对权重，点的权值随聚类进行而减小，可能产生更好的准确率和更快的收敛性
# 数据预处理
def pre(data, m, n, v):
    dataset = data.iloc[:, m:n]
    transfer = VarianceThreshold(threshold=v)
    data_dimension = pd.DataFrame(transfer.fit_transform(dataset))
    print(data_dimension)
    # 手肘法来获得合适的k
    sse(data_dimension)
    print_means(data_dimension, 6
                )


# iris 数据集
data = pd.read_csv("D:\桌面\数据集\数据集\iris.csv")
dataset = data.copy()
dataset.drop('label', axis=1, inplace=True)
print(dataset)

m = 1
n = 4
v = 0.4
pre(data, m, n, v)

#forestfire数据集
data = pd.read_csv("D:\桌面\数据集\数据集\\forestfires.csv")
m=9
n=10
v=0.9
k=5
pre(data,m,n,v)


#abalone数据集
data =pd.read_csv("D:\桌面\数据集\数据集\\abalone.csv")
m=4
n=8
v=0.03
pre(data,m,n,v)

#custom数据集
data = pd.read_csv("D:\桌面\数据集\数据集\custom.csv")
m=2
n=7
v=70000000
pre(data,m,n,v)

#hoilday

data = pd.read_csv("D:\桌面\数据集\数据集\holiday.csv")
m=2
n=7
v=1500
pre(data,m,n,v)
