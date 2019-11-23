import tensorflow as tf
from tensorflow.python.keras import layers, optimizers, datasets, Sequential
import tensorflow.keras as keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class BasicBlock(layers.Layer):
    def __init__(self,fliter_num,stride = 1):
        super(BasicBlock,self).__init__()
        #一个卷积层，后面再接一BN层，再加激活函数。
        self.conv1 = layers.Conv2D(fliter_num,kernel_size=[3,3],strides = stride,padding = 'same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        #第二个卷积层，接BN层，激活函数放在该层与短接层相加后
        self.conv2 = layers.Conv2D(fliter_num, kernel_size=[3, 3], strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        #如果步幅不为1，则做一个下采样。目的是保证短接层和两个卷积层输出的尺寸一致。
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(fliter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x
        #将定义好的层组合成一个block
    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        #将两个卷积层组合
        out = self.conv2(out)
        out = self.bn2(out)
        #定义短接层
        identity = self.downsample(inputs)
        #用Add将卷积层与短接层相加
        output = layers.add([out, identity])  
        output = tf.nn.relu(output)
        return output

    #构建resnet模块，将basicblock组合起来构成最终的层结构
class ResNet(keras.Model):
    #layer_dims是长度为4的一维数组，指示4个不同种类的basic block的个数。
    #我们的resnet34网络应当输入[3,4,6,3]
    #第二个参数是最终的输出，取决于原输出有几类。本例中为10
    def __init__(self, layer_dims, num_classes=10):
        super(ResNet, self).__init__()
        # 预处理层；一个63*3*3卷积层，加BN,relu激活再池化。
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
          # 创建4个Res Block
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self,input,training=None):
        x=self.stem(input)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        # [b,c]
        x=self.avgpool(x)
        x=self.fc(x)
        return x

    # 实现 Res Block； 创建一个Res Block
    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        # may down sample 也许进行下采样。
        # 对于当前Res Block中的Basic Block，我们要求每个Res Block只有一次下采样的能力。
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1)) 

        return res_blocks


# 数据预处理，仅仅是类型的转换。    [-1~1]
def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 0.5
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# 数据集的加载
(x,y),(x_test,y_test)=datasets.cifar10.load_data()
y = tf.squeeze(y)  # 或者tf.squeeze(y, axis=1)把1维度的squeeze掉。
y_test = tf.squeeze(y_test) 
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(64)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(64)

# 我们来测试一下sample的形状。
sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))  # 值范围为[0,1]


model = ResNet([2,2,2,2])
model.build(input_shape=(None, 32, 32, 3))
model.summary()
optimizer = tf.keras.optimizers.Adam(lr=1e-3)

for epoch in range(20):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            # [b, 32, 32, 3] => [b, 10]
            logits = model(x)
            # [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)
            # compute loss   结果维度[b]
            loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            loss = tf.reduce_mean(loss)

        # 梯度求解
        grads = tape.gradient(loss, model.trainable_variables)
        # 梯度更新
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 50 == 0:
            print(epoch, step, 'loss:', float(loss))

    # 做测试
    total_num = 0
    total_correct = 0
    for x, y in test_db:

        logits = model(x)
        # 预测可能性。
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)  # pred类型为int64,需要转换一下。
        pred = tf.cast(pred, dtype=tf.int32)

        # 拿到预测值pred和真实值比较。
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_num += x.shape[0]
        total_correct += int(correct)  # 转换为numpy数据

    acc = total_correct / total_num
    print(epoch, 'acc:', acc)

#保存模型权重
model.save_weights('F:/save_weights/weights.ckpt')
#想读取权重，要建立一个结构一样的，然后model.load_weights('F:/save_weights/weights.ckpt')