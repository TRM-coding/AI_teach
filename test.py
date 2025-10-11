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


SIZE = 256
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=10,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas')

canvas_result2 = st_canvas(
    fill_color='#000000',
    stroke_width=10,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas_2')