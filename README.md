### 环境需求

1. Python 3
2. OpenCV 3 w/ Python extensions
3. 在 requirements.txt 中的库
 - Run"pip3 install -r requirements.txt"

### Step 1: 将验证码图片切分成单个字符

Run:

python3 extract_single_letters_from_captchas.py

结果存在 "extracted_letter_images" 文件夹


### Step 2: 训练模型

Run:

python3 train_model.py

会在根目录下生成 "captcha_model.hdf5" and "model_labels.dat"


### Step 3: 使用模型解决问题

Run: 

python3 solve_captchas_with_model.py