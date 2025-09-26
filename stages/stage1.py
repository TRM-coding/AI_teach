# 导入 Streamlit 库，用于快速构建交互式 Web 应用
import streamlit as st

# 导入 Python 的垃圾回收模块
# gc 可以帮助释放不再使用的内存，防止长时间运行时内存占用过高
import gc
gc.enable()  # 启用自动垃圾回收机制（默认是启用的，这里显式写出来便于教学）


# ---------------- Streamlit 页面配置 ----------------
# st.set_page_config 用于设置整个应用的全局配置
# 参数：
# - page_title: 浏览器标签页上的标题
# - initial_sidebar_state: 设置侧边栏初始是否展开 ('expanded'/'collapsed')
st.set_page_config(page_title="Digit Recognizer App", initial_sidebar_state="expanded")


# ---------------- 页面内容 ----------------
# st.title: 显示一个大标题
st.title('Digit Recognizer')

# st.write: 可以输出文字、数据、甚至是图表，功能类似 print，但更智能
st.write("This is a simple image classification web app to **recognize the digit** drawn in the canvas.")



# # 1. 输出普通文本
# st.write("Hello, Streamlit!")

# # 2. 输出数字和表达式结果
# st.write(2 + 3)  # 会显示 5
# st.write("平方结果:", 7 ** 2)

# # 3. 输出 Python 列表或字典
# st.write([1, 2, 3, 4])  
# st.write({"name": "Alice", "age": 20})

# # 4. 输出表格（Pandas DataFrame）
# import pandas as pd
# df = pd.DataFrame({"数字": [1, 2, 3], "平方": [1, 4, 9]})
# st.write(df)

# # 5. 输出图表（Matplotlib / Plotly 都支持）
# import matplotlib.pyplot as plt
# import numpy as np
# x = np.linspace(0, 2*np.pi, 100)
# y = np.sin(x)
# fig, ax = plt.subplots()
# ax.plot(x, y)
# st.write(fig)  # 直接显示图像

# # 6. 混合文字 + 变量
# name = "小明"
# score = 95
# st.write(f"学生 {name} 的分数是 {score}")


# st.markdown: 用 Markdown 语法渲染文本，可以加粗、标题、列表等
st.markdown('### Step 1: 页面框架完成')




