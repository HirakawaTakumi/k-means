# encoding: shift_jis
import numpy
import random
import math
import pylab
import matplotlib.pyplot as plt

# グラフに描画
def draw_data( data, classes, centers ,colors = ("r" , "b" , "g" , "c" , "m" , "y" )):
    # データを描画
    for d,k in zip(data, classes):
        pylab.scatter( d[0] , d[1] , color=colors[k] )

    # クラスタ中心を描画
    for i,c in enumerate(centers):
        pylab.scatter( c[0] , c[1] , color=colors[i] , marker="x" )

def draw_line( p1 , p2 , color="k" ):
    pylab.plot( [p1[0], p2[0]] , [p1[1],p2[1]], color=color )

def plot1(K, d, data, classes, centers, errors ):
    pylab.clf()
    pylab.subplot( "121" )
    draw_data( data, classes, centers )
    for k in range(K):
        draw_line( d , centers[k] )
    pylab.subplot( "122" )
    pylab.plot( range(len(errors)) , errors )
    pylab.draw()
    pylab.pause(0.1)

def plot2(k_new, d, data, classes, centers, errors ):
    pylab.clf()
    pylab.subplot( "121" )
    draw_data( data, classes, centers )
    draw_line( d , centers[k_new] , "y" )
    pylab.subplot( "122" )
    pylab.plot( range(len(errors)) , errors )
    pylab.draw()
    pylab.pause(0.1)


def plot3(K, d, data, classes, centers, errors ):
    pylab.ioff()
    pylab.clf()
    pylab.subplot( "121" )
    draw_data( data, classes, centers )
    for d,c in zip(data,classes):
        draw_line( d , centers[c] )
    pylab.subplot( "122" )
    pylab.plot( range(len(errors)) , errors )
    pylab.show()

def calc_center(k, data, classes):
    center = numpy.zeros( len(data[0]) )
    n = 0

    # クラスkが割り当てられてるデータの総和と個数を計算
    for d,c in zip(data,classes):
        if c==k:
            center += d
            n += 1

    # 平均を計算
    return center / n

def find_nearest_class( d , centers ):
    dists = []
    for c in centers:
        # 距離を計算
        dists.append( numpy.linalg.norm( d - c ) )

    # 最小距離のクラスを返す
    return numpy.argmin( dists )

def calc_error( data , classes, centers ):
    err = 0
    for i in range(len(data)):
        d = data[i]
        k = classes[i]
        c = centers[k]
        err += numpy.linalg.norm( d - c )
    return err



# k-meansメイン
def kmeans( data , K ):
    pylab.ion()

    # データの次元
    dim = len(data[0])

    # データをランダムに分類
    classes = numpy.random.randint( K , size=len(data) )

    # クラスタの中心を計算
    centers = [ None for c in range(K) ]
    for k in range(K):
        centers[k] = calc_center( k , data , classes )

    # グラフ表示
    draw_data( data, classes, centers )
    pylab.draw()
    pylab.pause(1.0)

    # 量子化誤差
    errors = []

    for it in range(1):
        # メインの処理
        for i in range(len(data)):
            d = data[i]
            k_old = classes[i]  # 現在のクラス

            # グラフ表示
            plot1(K, d, data, classes, centers, errors )

            # 最近傍のクラスを見つける
            k_new = find_nearest_class( d , centers )

            # 分類に変化があったらクラスタ中心を更新
            if k_old!=k_new:
                classes[i] = k_new
                centers[k_old] = calc_center( k_old , data , classes )
                centers[k_new] = calc_center( k_new , data , classes )

            # グラフ表示
            plot2(k_new, d, data, classes, centers, errors )

            e = calc_error( data, classes, centers )
            errors.append(e)
            print "量子化誤差：",e


    # 最終的な結果を表示
    plot3(K, d, data, classes, centers, errors )

def main():
    data = numpy.loadtxt( "data1.txt" )
    kmeans( data , 2 )

if __name__ == '__main__':
    main()