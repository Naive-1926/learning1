import tensorflow as tf
from tensorflow import keras
import numpy as np

#先定义一个类用来导入MNIST数组
class Mnistloader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data,self.train_label),(self.test_data,self.test_label) = mnist.load_data()  #导入MNIST数据集
        #对数据进行格式转换，原格式是Unit8,tensorflow的Api不认
        self.train_data = self.train_data.astype(np.float32)/255.0
        self.test_data = self.test_data.astype(np.float32)/255.0
        self.train_label = self.train_label.astype(np.int32)
        self.test_label = self.test_label.astype(np.int32)
        #统计训练集和测试集的样本总数
        self.num_train = self.train_data.shape[0]
        self.num_test = self.test_data.shape[0]
        #该方法从总集中随机选出size个数据（随机抽取器）
    def get(self,size):
        index = np.random.randint(0,np.shape(self.train_data)[0],size)
        train_data_apply = self.train_data
        train_data_apply = train_data_apply[index,:]
        train_label_apply = self.train_label
        train_label_apply = train_label_apply[index]
        return train_data_apply , train_label_apply

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        #定义接下来要用到的层数
        self.flatten = tf.keras.layers.Flatten()  #该层用于将输入层展平
        #全连接层
        self.dense1 = tf.keras.layers.Dense(units=1024,activation=tf.nn.relu)   #第一隐藏层有1024个节点，使用relu进行非线性激活
        self.dense2 = tf.keras.layers.Dense(units=512,activation=tf.nn.relu)    #第二隐藏层有512个节点
        self.dense3 = tf.keras.layers.Dense(units=10)                           #输出层共10个输出
        #call函数定义了参数在不同的层中传播的过程
    def call(self,input):
        x = self.flatten(input)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        output = tf.nn.softmax(x)
        return output

#训练模型
epoch_num = 3      #训练轮数为5轮
batch_size = 50    #每批次训练个数
learning_rate = 0.001   #学习率
data_loader = Mnistloader()   #实例化Mnist载入器
model = MLP()                 #实例化训练模型
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)  #实例化优化器
num_batchs = int(data_loader.num_train//batch_size*epoch_num)        #定义共需喂数据的次数

for batch in range(num_batchs):           #一共喂数据num_batchs次
    X,y = data_loader.get(batch_size)     #每次给DNN喂入batch_size个数据
    #在GradienTape中可查看梯度
    with tf.GradientTape() as Tape:
        #计算预测值
        y_pred = model(X)  
        #计算损失函数，使用交叉熵函数
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)  
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch , loss.numpy()))
    #计算损失函数的导数,model.variables可以传入model模型的所有权重
    grads = Tape.gradient(loss,model.variables)
    #使用优化器更新权重
    optimizer.apply_gradients(grads_and_vars = zip(grads,model.variables))

#定义评估器
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
#定义迭代轮数
num_test_batches = int(data_loader.num_test//batch_size)
    #分批次进行迭代
for batch_index in range(num_test_batches):
    start = batch_index * batch_size
    end = (batch_index+1)*batch_size #每个批次所处理的测试样本的位置
    #进行预测
    y_test_pred = model.predict(data_loader.test_data[start:end])
    sparse_categorical_accuracy.update_state(y_true = data_loader.test_label[start:end],y_pred = y_test_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())