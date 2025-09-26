# 导入 Streamlit 库
import streamlit as st

# 导入 Streamlit 提供的画布组件（第三方库）
# st_canvas 可以让用户在网页上绘制图像
from streamlit_drawable_canvas import st_canvas

# 导入垃圾回收模块
import gc
gc.enable()  # 显式启用垃圾回收


# ---------------- Streamlit 页面配置 ----------------
# 设置网页的全局配置
st.set_page_config(
    page_title="Digit Recognizer App",       # 浏览器标签页的名字
    initial_sidebar_state="expanded"         # 侧边栏默认展开
)

# ---------------- 页面内容 ----------------
# 页面标题
st.title('Digit Recognizer')

# 输出提示文字
st.write("画布已经可以绘制数字了")

# ---------------- 侧边栏配置 ----------------
# 在侧边栏显示一个小标题
st.sidebar.header("Configuration")

# st.sidebar.slider: 添加一个滑动条，让用户调节笔刷宽度
# 参数依次为 (标签, 最小值, 最大值, 默认值)
stroke_width = st.sidebar.slider("Brush width: ", 10, 30, 20)

# st.sidebar.checkbox: 添加一个勾选框，选择是否启用绘画模式
# 如果勾选，返回 True；否则返回 False
drawing_mode = st.sidebar.checkbox("Drawing mode ?", True)


# ---------------- 画布组件 ----------------
# 定义画布大小
SIZE = 256

# st_canvas: 提供一个可以画图的区域
canvas_result = st_canvas(
    fill_color='#000000',                 # 填充颜色（黑色，透明度0）
    stroke_width=stroke_width,            # 笔刷宽度（来自滑动条）
    stroke_color='#FFFFFF',               # 笔刷颜色（白色）
    background_color='#000000',           # 背景颜色（黑色）
    width=SIZE,                           # 画布宽度
    height=SIZE,                          # 画布高度
    drawing_mode="freedraw" if drawing_mode else "transform",
    # drawing_mode: 绘图模式
    # - "freedraw": 自由绘制模式
    # - "transform": 拖动/缩放图像模式

    key='canvas'                          # 组件的唯一标识符
)

canvas_result2 = st_canvas(
    fill_color='#000000',                 # 填充颜色（黑色，透明度0）
    stroke_width=stroke_width,            # 笔刷宽度（来自滑动条）
    stroke_color='#FFFFFF',               # 笔刷颜色（白色）
    background_color='#000000',           # 背景颜色（黑色）
    width=SIZE,                           # 画布宽度
    height=SIZE,                          # 画布高度
    drawing_mode="freedraw" if drawing_mode else "transform",
    # drawing_mode: 绘图模式
    # - "freedraw": 自由绘制模式
    # - "transform": 拖动/缩放图像模式

    key='canvas2'                          # 组件的唯一标识符
)

# 运行后，网页会显示一个黑色背景的画布，可以用鼠标画白色的线条




import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Two Canvas Example", layout="wide")
st.title("两个画布并排")

# 定义画布大小
SIZE = 200

# 创建两列
col1, col2 ,col3 = st.columns(3)

with col1:
    st.subheader("画布 A")
    canvas_result_A = st_canvas(
        fill_color="#000000",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw",
        key="canvasA"
    )

with col2:
    st.subheader("画布 B")
    canvas_result_B = st_canvas(
        fill_color="#000000",
        stroke_width=15,
        stroke_color="#FF0000",  # 红色画笔
        background_color="#000000",
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw",
        key="canvasB"
    )

with col3:
    st.subheader("画布 C")
    canvas_result_B = st_canvas(
        fill_color="#000000",
        stroke_width=15,
        stroke_color="#FF0000",  # 红色画笔
        background_color="#000000",
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw",
        key="canvasC"
    )