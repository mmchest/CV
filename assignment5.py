
# k - means ++

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

def assignment(df, centroids, colmap):    # (distance)
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )

    # 接下来是要给点上色
    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])   # 上色
    return df


def nearest(df, centroids):
    min_dist = 10000000;                   # 随便给的，一开始尽量大
    for i in centroids.keys():            # 计算point与每个聚类中心之间的距离
        if min_dist > df['distance'][i]:
            min_dist = df['distance'][i]
    return min_dist


# 选择尽可能相距较远的类中心
def get_centroids(df, centroids, k):
    m = len(df['x'])
    index = np.random.randint(0, m)
    centroids[index][0] = df[index]['x']
    centroids[index][1] = df[index]['y']

    d = [0.0 for _ in range(m)]
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            d[j] = nearest(df, centroids)
            sum_all += d[j]
        sum_all *= np.random.rand()
        for j, di in enumerate(d):
            sum_all=sum_all - di
            if sum_all > 0:
                continue
            centroids[i][0] = df[j]['x']
            centroids[i][1] = df[j]['y']
            break
    return centroids


def main():
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })
    # dataframe 返回一个二维矩阵，用.loc直接定位
    # 看明白了，暂时还没体会这种用法的方便之处…

    k = 3
    # centroids[i] = [x, y]
    centroids = {
        i: [np.random.randint(0, 80), np.random.randint(0, 80)]
        for i in range(k)
    }
    # 可以认为centroids是个字典。这句话是给字典赋值，key是i，value是一个数组，里面是两个随机数。
    # 这一步先产生三个聚类的中心点


    colmap = {0: 'r', 1: 'g', 2: 'b'}            # 给不同的聚类上色
    df = assignment(df, centroids, colmap)       # 求距离

    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i], linewidths=6)       # 默认color = none，所以重新赋值
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()            # 接下来的for循环就是出图

    for i in range(10):
        key = cv2.waitKey()
        plt.close()

        closest_centroids = df['closest'].copy(deep=True)
        centroids = get_centroids(df, centroids, k)

        plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.show()

        df = assignment(df, centroids, colmap)

        if closest_centroids.equals(df['closest']):           # 不变了就不画图了
            break


if __name__ == '__main__':          # 程序入口
    main()