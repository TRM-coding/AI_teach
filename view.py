import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import SimpleDNN as Model  # 假设模型在model.py中定义
from utils import load_data, train_model, evaluate_model

st.set_page_config(page_title="MNIST训练可视化", layout="wide")
st.title("🧠 MNIST数字识别模型训练可视化")

# 初始化session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'train_loader' not in st.session_state:
    st.session_state.train_loader = None
if 'test_loader' not in st.session_state:
    st.session_state.test_loader = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'history' not in st.session_state:
    st.session_state.history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

# 侧边栏配置
st.sidebar.header("⚙️ 训练配置")
batch_size = st.sidebar.number_input("Batch Size", min_value=16, max_value=512, value=64, step=16)
learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
num_epochs = st.sidebar.number_input("训练轮数", min_value=1, max_value=100, value=10, step=1)

# 定义训练历史可视化函数（移到前面）
def create_training_plots(history):
    """创建训练历史可视化图表"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss曲线', 'Accuracy曲线')
    )
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Loss曲线
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', mode='lines+markers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['test_loss'], name='Test Loss', mode='lines+markers'),
        row=1, col=1
    )
    
    # Accuracy曲线
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc', mode='lines+markers'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['test_acc'], name='Test Acc', mode='lines+markers'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True)
    return fig

# 数据加载部分
st.header("📊 数据管理")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 加载数据", type="primary", use_container_width=True):
        with st.spinner("正在加载数据到GPU..."):
            try:
                train_loader, test_loader, device,train_size,test_size = load_data(batch_size)
                st.session_state.train_loader = train_loader
                st.session_state.test_loader = test_loader
                st.session_state.device = device
                st.session_state.data_loaded = True
                
                # 初始化模型
                st.session_state.model = Model().to(device)
                
                st.success(f"✅ 数据加载成功！使用设备: {device}")
                st.info(f"训练样本数: {train_size} | 测试样本数: {test_size}")
            except Exception as e:
                st.error(f"❌ 数据加载失败: {str(e)}")

with col2:
    if st.button("🔃 重新加载数据", use_container_width=True):
        st.session_state.data_loaded = False
        st.session_state.model = None
        st.session_state.history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        st.success("✅ 数据已清除，请重新加载")
        st.rerun()

# 训练和评估部分
st.header("🎯 模型训练与评估")
col3, col4 = st.columns(2)

with col3:
    if st.button("▶️ 开始训练", type="primary", use_container_width=True, disabled=not st.session_state.data_loaded):
        if st.session_state.data_loaded:
            model = st.session_state.model
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # progress_bar:进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            # 占位符，防止图表重复渲染，造成若干图表叠加
            chart_placeholder = st.empty()
            
            # 定义回调函数用于更新UI
            def update_progress(epoch, train_loss, train_acc, test_loss, test_acc):
                # 更新历史记录
                st.session_state.history['train_loss'].append(train_loss)
                st.session_state.history['train_acc'].append(train_acc)
                st.session_state.history['test_loss'].append(test_loss)
                st.session_state.history['test_acc'].append(test_acc)
                
                # 更新进度条
                progress_bar.progress(epoch / num_epochs)
                status_text.text(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
                
                # 更新图表 - 添加唯一key
                fig = create_training_plots(st.session_state.history)
                chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"training_chart_{epoch}")
            
            # 开始训练
            with st.spinner("训练中..."):
                try:
                    history = train_model(
                        model, 
                        st.session_state.train_loader, 
                        st.session_state.test_loader,
                        criterion, 
                        optimizer, 
                        num_epochs, 
                        st.session_state.device,
                        progress_callback=update_progress
                    )
                    st.success("🎉 训练完成！")
                except Exception as e:
                    st.error(f"❌ 训练失败: {str(e)}")

with col4:
    if st.button("📈 评估模型", use_container_width=True, disabled=not st.session_state.data_loaded):
        if st.session_state.model is not None:
            with st.spinner("评估中..."):
                try:
                    criterion = nn.CrossEntropyLoss()
                    test_loss, test_acc = evaluate_model(
                        st.session_state.model,
                        st.session_state.test_loader,
                        criterion,
                        st.session_state.device
                    )
                    
                    st.metric("测试集Loss", f"{test_loss:.4f}")
                    st.metric("测试集准确率", f"{test_acc:.2f}%")
                except Exception as e:
                    st.error(f"❌ 评估失败: {str(e)}")

# 历史训练数据展示
st.header("📉 训练历史")

if len(st.session_state.history['train_loss']) > 0:
    # 显示图表 - 添加唯一key
    fig = create_training_plots(st.session_state.history)
    st.plotly_chart(fig, use_container_width=True, key="history_chart")
    
    # 显示数据表格
    st.subheader("📋 详细数据")
    df = pd.DataFrame({
        'Epoch': range(1, len(st.session_state.history['train_loss']) + 1),
        'Train Loss': st.session_state.history['train_loss'],
        'Train Acc (%)': st.session_state.history['train_acc'],
        'Test Loss': st.session_state.history['test_loss'],
        'Test Acc (%)': st.session_state.history['test_acc']
    })
    st.dataframe(df, use_container_width=True)
else:
    st.info("📭 暂无训练历史，请先加载数据并开始训练")