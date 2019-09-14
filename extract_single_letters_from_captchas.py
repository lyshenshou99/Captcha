import os
import os.path
import cv2
import glob
import imutils


CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"


# 将所有需要处理的图片整理到一个列表中
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # 验证码文件名作为文本
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # 载入图片并转化为灰度图
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 填充
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # 将图片二阶化
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 将图片转化成4个部分
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 兼容性！！！！
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    for contour in contours:
        # 切成矩形
        (x, y, w, h) = cv2.boundingRect(contour)

        # 以长宽比判断是否有粘连
        if w / h > 1.25:
            # 有粘连就对半切开
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))

    # 不是4个字符的验证码不算在失败模型中！
    if len(letter_image_regions) != 4:
        continue

    # 从左到右处理（基于X坐标）
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # 分开存储4个字符的单独图片
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # 获取字符坐标
        x, y, w, h = letter_bounding_box

        # 以2像素作为边距提取字符
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        # 获取保存图片的目录
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # 如果目标目录不存在就创建一个！
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 输出至文件中
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # 计数器增加
        counts[letter_text] = count + 1
