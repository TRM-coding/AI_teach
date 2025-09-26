# 导入 Streamlit 库
import streamlit as st

# 导入可绘制画布组件
from streamlit_drawable_canvas import st_canvas

# 导入 OpenCV，用于图像处理
import cv2

# 导入垃圾回收模块
import gc
gc.enable()  # 启用垃圾回收


# ---------------- 页面设置 ----------------
st.set_page_config(
    page_title="Digit Recognizer App",   # 浏览器标签标题
    initial_sidebar_state="expanded"     # 侧边栏默认展开
)

# 页面标题和说明
st.title('Digit Recognizer')
st.write("第三步：图像预处理，把绘制的数字转为模型输入格式。")


# ---------------- 画布设置 ----------------
SIZE = 256  # 画布显示的大小
canvas_result = st_canvas(
    fill_color='#000000',        # 填充颜色
    stroke_width=20,             # 画笔粗细
    stroke_color='#FFFFFF',      # 画笔颜色（白色）
    background_color='#000000',  # 背景颜色（黑色）
    width=SIZE,                  # 画布宽度
    height=SIZE,                 # 画布高度
    drawing_mode="freedraw",     # 自由绘制模式
    key='canvas'                 # 组件唯一标识符
)


# ---------------- 图像预处理 ----------------
if canvas_result.image_data is not None:
    # 1. 将绘制的图像缩放为 28x28
    # MNIST 数据集中的数字图像就是 28x28 尺寸
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))

    # 2. 为了更清楚展示模型输入，把 28x28 再放大显示为 256x256
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

    # 3. 在网页上显示放大的“模型输入图像”
    st.write('模型输入（28x28，灰度处理后）：')
    st.image(rescaled)

    # 4. 转换为灰度图（单通道），去掉彩色的三通道
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray=img
    # 5. 显示灰度图
    st.image(img, caption="Grayscale Input")




# 图像旋转

import cv2
import streamlit as st

# 假设 img_gray 是 28x28 的灰度图
angle = st.slider("选择旋转角度", -180, 180, 30)  # 用滑块选择角度
(h, w) = img_gray.shape[:2]  # 高度和宽度
center = (w // 2, h // 2)    # 旋转中心

# 获取旋转矩阵
M = cv2.getRotationMatrix2D(center, angle, 1.0)

# 仿射变换实现旋转
rotated = cv2.warpAffine(img_gray, M, (w, h))

st.image(rotated, caption=f"旋转 {angle}° 后的图像")


# 更改图像分辨率

# 分辨率选项
target_size = st.radio("选择目标分辨率", [(14, 14), (28, 28), (56, 56), (112, 112)])
resampled = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_NEAREST)

st.image(resampled, caption=f"分辨率 {target_size} 的图像")