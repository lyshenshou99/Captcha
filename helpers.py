import imutils
import cv2


def resize_to_fit(image, width, height):
    """
    调整图片大小的函数
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # 获取图像的原大小并填充
    (h, w) = image.shape[:2]

    # 宽度大于高度就以宽度为依据
    if w > h:
        image = imutils.resize(image, width=width)

    # 不然就以长度为依据调整
    else:
        image = imutils.resize(image, height=height)

    # 确定大小和宽度
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # 填充并调整大小
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # 返回修改后图像
    return image