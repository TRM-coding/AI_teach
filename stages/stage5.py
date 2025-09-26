# ---------------- 导入库 ----------------
import streamlit as st                       # Streamlit：搭建交互式 Web 界面
from streamlit_drawable_canvas import st_canvas  # 画布组件，用于手写输入
import cv2                                   # OpenCV：图像处理
import torch                                 # PyTorch：深度学习框架
import pandas as pd                          # Pandas：数据表格处理
import plotly.graph_objects as go            # Plotly：绘制交互式图表
from model import transform_image, predict_image  # 自定义的预处理和预测函数
import gc                                    # Python 垃圾回收模块

gc.enable()  # 显式开启垃圾回收


# ---------------- 页面设置 ----------------
st.set_page_config(page_title="Digit Recognizer App", initial_sidebar_state="expanded")
st.title('Digit Recognizer')
st.write("第五步：结果可视化。")


# ---------------- 定义绘图函数 ----------------
def plot_fig(df):
    """
    输入：一个 DataFrame（行是类别，列是概率）
    功能：绘制类别 vs 概率的水平条形图
    """
    fig = go.Figure(go.Bar(
        x=df[0].tolist(),        # x 轴：概率值
        y=list(df.index),        # y 轴：类别（数字 0-9）
        orientation='h'))        # orientation='h' 表示横向柱状图

    fig.update_yaxes(type='category')  # y 轴显示为分类类别，而不是数值
    st.plotly_chart(fig)               # 在 Streamlit 页面上展示图表


# ---------------- 绘图区域（画布） ----------------
SIZE = 256
canvas_result = st_canvas(
    fill_color='#000000',        # 填充颜色（黑色）
    stroke_width=20,             # 笔刷宽度
    stroke_color='#FFFFFF',      # 笔刷颜色（白色）
    background_color='#000000',  # 背景颜色（黑色）
    width=SIZE,                  # 画布宽度
    height=SIZE,                 # 画布高度
    drawing_mode="freedraw",     # 绘画模式：自由绘制
    key='canvas'                 # 组件唯一标识符
)

prob=None
# ---------------- 模型预测与可视化 ----------------
if canvas_result.image_data is not None:
    # 1. 图像预处理
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))  # 缩放至28x28
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                      # 转灰度图
    tensor_img = transform_image(img_gray)                                # 转换为 PyTorch Tensor

    # 2. 点击按钮后进行预测
    if st.button("Run Prediction"):
        class_label, confidence, probss = predict_image(tensor_img)
        prob=probss
        # 显示预测结果
        st.write("预测类别：", class_label)
        st.write("置信度：", f"{confidence}%")

        # 3. 用 Pandas 显示概率分布
        df = pd.DataFrame(probss.numpy()).transpose()
        st.dataframe(df)  # 在网页上展示数据表格

        # 4. 用 Plotly 绘制条形图
        plot_fig(df)


#-------------------------------------------------基于matplotlib的绘图#

#饼图
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
classes = list(range(10))                # 类别 0-9
ax.bar(classes, prob.numpy())           # y: 概率
ax.set_xlabel("Classes")
ax.set_ylabel("Probability")
ax.set_title("概率分布（Bar Chart）")
st.pyplot(fig)                           # 用 Streamlit 显示 Matplotlib 图

#折线图

fig, ax = plt.subplots()
ax.plot(classes, prob.numpy(), marker='o')
ax.set_xlabel("Classes")
ax.set_ylabel("Probability")
ax.set_title("概率分布（Line Plot）")
st.pyplot(fig)







#饼图

import plotly.express as px

# 把概率转成 DataFrame
df = pd.DataFrame({
    "class": list(range(10)),
    "prob": prob.numpy().tolist()
})

# 绘制饼图
fig = px.pie(df, values="prob", names="class", title="概率分布 (Pie Chart)")
st.plotly_chart(fig)


# 折线图
fig = px.line(df, x="class", y="prob", markers=True, title="概率分布 (Line Chart)")
st.plotly_chart(fig)

# 热力图
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sns.heatmap([prob.numpy()], annot=True, cmap="Blues", cbar=True, ax=ax,
            xticklabels=list(range(10)), yticklabels=["Prob"])
st.pyplot(fig)

#高亮预测

st.markdown(
    f"<h2 style='text-align:center; color:green;'>预测结果：{class_label} （{confidence}%）</h2>",
    unsafe_allow_html=True
)

