import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit


LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"


# 初始化数据和标签
data = []
labels = []

# 处理输入的图像
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # 载入图片并转换成灰度图
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 调整图片的大小至20*20
    image = resize_to_fit(image, 20, 20)

    # 为图像添加第三通道 让Keras工作
    image = np.expand_dims(image, axis=2)

    # 给标签命名（根据文件夹）
    label = image_file.split(os.path.sep)[-2]

    # 将训练集和标签导入模型中
    data.append(image)
    labels.append(label)


# 为加快训练效率 将image图形二阶化
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# 将训练数据拆分为单独的训练集和测试集
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# 因为要使用Keras 将图像和标签转换成one-hot编码
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# 保存从标签到one-hot编码的映射
# 用这个映射来转换后面的one-hot编码到标签
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# 搭建神经网络框架
model = Sequential()

# CNN中第一个具有最大池的卷积层
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 第二个具有最大池的卷积层
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 隐藏层设置500个节点
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# 输出层设置32个节点
model.add(Dense(32, activation="softmax"))

# 用Keras在后台搭建TensorFlow模型
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 开始训练
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)

# 把训练结果保存到文件中
model.save(MODEL_FILENAME)
