from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"


# 加载标签
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# 加载训练后的神经网络模型
model = load_model(MODEL_FILENAME)

# 随机抓取一些训练集中的数据测试一下
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)


for image_file in captcha_image_files:
    # 加载图像并转换成灰度图
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 填充图片至大小为20*20
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # 将图像二阶化
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 以连续的像素点作为图像的轮廓
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 尽量兼容所有版本的opencv
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    # 遍历4个轮廓中的每一个然后识别
    for contour in contours:
        # 按照轮廓切成矩形
        (x, y, w, h) = cv2.boundingRect(contour)

        # 防止轮廓有粘连 检查长宽比
        if w / h > 1.25:
            # 符合这个条件估计就是两个粘到一起 从中间分开
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))

    # 跳过不是4个字符的验证码
    if len(letter_image_regions) != 4:
        continue

    # 从左到右处理（以X坐标）
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # 创建输出图像和存放识别结果的列表
    output = cv2.merge([image] * 3)
    predictions = []

    for letter_bounding_box in letter_image_regions:
        # 获取图像中字符元素的坐标
        x, y, w, h = letter_bounding_box

        # 以2像素作为边距提取字符
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

        # 将图像大小转换成20*20
        letter_image = resize_to_fit(letter_image, 20, 20)

        # 给图像添加到4通道来用 Keras
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # 应用神经网络模型来识别结果
        prediction = model.predict(letter_image)

        # 把推测出来的one-hot编码转换成标签
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        # 在输出图像上输出结果
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # 输出验证码的文字
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))

    # 输出图像
    cv2.imshow("Output", output)
    cv2.waitKey()