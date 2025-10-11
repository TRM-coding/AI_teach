import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import torch
import pandas as pd
import plotly.graph_objects as go
from model import transform_image, predict_image
import gc

gc.enable()

st.set_page_config(page_title="Digit Recognizer App", initial_sidebar_state="expanded")
st.title('Digit Recognizer')
st.write("最终阶段：完整数字识别应用。")

def plot_fig(df):
    fig = go.Figure(go.Bar(
        x=df[0].tolist(),
        y=list(df.index),
        orientation='h'))
    fig.update_yaxes(type='category')
    fig.update_layout(
        title={'text': "Classes vs Probabilities", 'x':0.5, 'xanchor': 'center'},
        xaxis_title="Probabilities",
        yaxis_title="Classes")
    st.plotly_chart(fig)

SIZE = 256
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas'
)

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.image(rescaled, caption="模型输入（28x28缩放）")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if st.button("Run Prediction"):
        with st.spinner("模型推理中..."):
            tensor_img = transform_image(img_gray)
            class_label, confidence, probs = predict_image(tensor_img)
        print(probs)
        if confidence > 50:
            st.success("预测成功！")
        else:
            st.warning("预测不确定")

        st.write("预测类别：", class_label)
        st.write("置信度：", f"{confidence}%")

        df = pd.DataFrame(probs.numpy()).transpose()
        st.dataframe(df)
        plot_fig(df)

gc.collect()
