from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import matplotlib.pyplot as plt


    #导入MNIST数据集
mnist = tf.keras.datasets.mnist
    #获取训练集和测试集
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()  

#plt.imshow(X_train[0],cmap=plt.cm.binary)显示了第一张图像,证明载入正确
#plt.show() 

#预处理
#格式转换，便于距离运算
X_TRA_FLOAT = tf.cast(X_train,dtype = tf.float32)
Y_TRA_INT = tf.cast(Y_train,dtype = tf.int32)
X_TES_FLOAT = tf.cast(X_test,dtype = tf.float32)
Y_TES_INT = tf.cast(Y_test,dtype = tf.int32)
#将图片集降维，即把每个图片矩阵展开成一维数组
X_TRA_APPLY = tf.reshape(X_TRA_FLOAT,[-1,28*28])
X_TES_APPLY = tf.reshape(X_TES_FLOAT,[-1,28*28])

#先计算向量距离，然后把与该Test向量距离最近的一个训练集中的向量找出来，返回其标签
def Min_Distance_Index(test):
    dis = tf.reduce_sum(tf.abs(tf.subtract(X_TRA_APPLY,test)),axis = 1)
    pred = tf.argmin(dis,0)
    return pred

Succeed = 0.   #用于计算准确率的计数器
Epoc = 200     #预测的向量数目（10000个全预测完真的慢，这是KNN的缺点，测试样本越多越慢）

#开始预测
for i in range(Epoc):
    Test_Index = Min_Distance_Index(X_TES_APPLY[i])    #获取最近邻向量的标签
    print("test:",i," prediction:",Y_TRA_INT[Test_Index].numpy()," true class",Y_TES_INT[i].numpy())   
    if Y_TRA_INT[Test_Index].numpy() == Y_TES_INT[i].numpy():  #最近邻向量的类别和真实的类别比较，类别一致则预测成功
        Succeed += 1.
Accuracy = Succeed/Epoc
print("The accuracy of the algorithm is:",Accuracy)  #打印最终的准确率




    