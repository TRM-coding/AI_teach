# 导入必要的库
import streamlit as st                      # Streamlit: 用于快速创建交互式Web应用
from streamlit_drawable_canvas import st_canvas  # st_canvas: 提供网页上的画布组件
import cv2                                  # OpenCV: 图像处理库
import torch                                # PyTorch: 深度学习框架
from model import transform_image, predict_image  # 自定义的函数, 需要提前写好
import gc                                   # 垃圾回收模块

gc.enable()  # 显式启用垃圾回收机制，防止长时间运行时内存占用过高


# ---------------- 页面设置 ----------------
st.set_page_config(page_title="Digit Recognizer App", initial_sidebar_state="expanded")

# 页面标题
st.title('Digit Recognizer')

# 页面说明
st.write("第四步：模型预测。")


# ---------------- 绘图区域 ----------------
SIZE = 256
canvas_result = st_canvas(
    fill_color='#000000',        # 填充颜色（黑色透明）
    stroke_width=20,             # 笔刷宽度（粗细）
    stroke_color='#FFFFFF',      # 笔刷颜色（白色）
    background_color='#000000',  # 背景颜色（黑色）
    width=SIZE,                  # 画布宽度
    height=SIZE,                 # 画布高度
    drawing_mode="freedraw",     # 绘图模式：自由绘制
    key='canvas'                 # 组件的唯一标识符
)


# ---------------- 图像预处理 ----------------
if canvas_result.image_data is not None:   # 判断用户是否已经画了内容
    # 1. 将画布的图像缩放为 28x28 像素（MNIST标准输入大小）
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))

    # 2. 转换为灰度图（单通道），简化输入
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 调用自定义函数，把灰度图转为 PyTorch Tensor（模型输入格式）
    tensor_img = transform_image(img_gray)


    # ---------------- 模型预测 ----------------
    # 创建一个按钮，点击后执行预测
    clicked = st.button("Run Prediction")

    if clicked:
        # 调用自定义函数进行预测
        class_label, confidence, probs = predict_image(tensor_img)

        # 输出提示框，表示运行完成
        st.success("运行完成")

        # 输出预测结果
        st.write("预测类别：", class_label)
        st.write("置信度：", f"{confidence}%")
