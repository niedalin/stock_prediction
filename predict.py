import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="股票价格预测系统", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #0F4C81;
    }
    h1, h2, h3, h4, span {
        color: #0F4C81 !important;
    }
    div[data-baseweb="input"] {
        border: 1px solid #0F4C81;
    }
    .css-1r6slb0, .css-1v3fvcr {
        background-color: #F0F8FF;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 股票价格预测系统 (LSTM vs Decision Tree)")

st.markdown("### 选择股票")
ticker = st.text_input("请输入股票代码 (如: 600578)", value="600578")

file_path = None
stock_dir = "stock"
if ticker.startswith('sh') or ticker.startswith('sz'):
    file_path = f"{stock_dir}/{ticker}.csv"
else:
    for f in os.listdir(stock_dir):
        if f.endswith('.csv') and ticker in f:
            file_path = f"{stock_dir}/{f}"
            break

if file_path is None or not os.path.exists(file_path):
    st.error(f"找不到数据文件: stock/{ticker}*.csv")
    st.stop()

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    return df

df = load_data(file_path)

with st.expander("数据特征与预处理 (Data Discovery & Preprocessing)", expanded=False):
    st.write("**数据描述性统计 (Descriptive Statistics):**")
    st.dataframe(df['close'].describe())
    
    missing_count = df.isnull().sum().sum()
    st.write(f"**数据清洗:** 检测到 {missing_count} 个缺失值。使用前向填充 (ffill) 处理。")
    df = df.ffill().bfill()
    
    prices = df[['close']].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)
    st.write("**数据归一化:** 使用 MinMaxScaler 将收盘价缩放至 [0, 1] 区间。")

SEQ_LEN = 30

X, y, target_dates = [], [], []
dates = df.index

for i in range(len(scaled_data) - SEQ_LEN):
    X.append(scaled_data[i : i + SEQ_LEN])
    y.append(scaled_data[i + SEQ_LEN])
    target_dates.append(dates[i + SEQ_LEN])

X = np.array(X)
y = np.array(y)
target_dates = pd.to_datetime(target_dates)

train_mask = target_dates.year <= 2023
test_mask = target_dates.year >= 2024

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
dates_test = target_dates[test_mask]

if len(X_test) == 0:
    st.error("数据集中没有2024年以后的数据用于测试！")
    st.stop()

st.markdown("###  模型预测")

@st.cache_resource
def train_lstm(X_train, y_train):
    model = Sequential()
    model.add(LSTM(8, return_sequences=True, input_shape=(SEQ_LEN, 1)))
    model.add(LSTM(8, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return model

with st.spinner("正在训练 LSTM 模型..."):
    lstm_model = train_lstm(X_train, y_train)
    lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)

X_train_dt = X_train.reshape(X_train.shape[0], -1)
X_test_dt = X_test.reshape(X_test.shape[0], -1)

n_features = 10
X_train_dt = X_train_dt[:, -n_features:]
X_test_dt = X_test_dt[:, -n_features:]

@st.cache_resource
def train_dt(X_train_dt, y_train):
    dt = DecisionTreeRegressor(
        max_depth=20, 
        min_samples_leaf=5,
        min_samples_split=10
    )
    dt.fit(X_train_dt, y_train.ravel())
    return dt

with st.spinner("正在训练 Decision Tree 模型..."):
    dt_model = train_dt(X_train_dt, y_train)
    dt_pred_scaled = dt_model.predict(X_test_dt).reshape(-1, 1)

y_test_real = scaler.inverse_transform(y_test)
lstm_pred_real = scaler.inverse_transform(lstm_pred_scaled)
dt_pred_real = scaler.inverse_transform(dt_pred_scaled)

def plot_prediction(dates, y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.plot(dates, y_true, color='black', label='Real Data', linewidth=1.5)
    ax.plot(dates, y_pred, color='red', label='Predicted Data', linewidth=1.5)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, color='#0F4C81', fontweight='bold', fontsize=10)
    ax.tick_params(colors='#0F4C81', axis='x', rotation=45, labelsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    fig.autofmt_xdate(rotation=45, ha='right')
    return fig

col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_prediction(dates_test, y_test_real, lstm_pred_real, "LSTM Prediction (2024 - Present)"))

with col2:
    st.pyplot(plot_prediction(dates_test, y_test_real, dt_pred_real, "Decision Tree Prediction (2024 - Present)"))

st.markdown("---")
st.markdown("###  模型评估与对比 (Model Evaluation)")

lstm_mse = mean_squared_error(y_test_real, lstm_pred_real)
lstm_mape = mean_absolute_percentage_error(y_test_real, lstm_pred_real) * 100

dt_mse = mean_squared_error(y_test_real, dt_pred_real)
dt_mape = mean_absolute_percentage_error(y_test_real, dt_pred_real) * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("LSTM 均方误差 (MSE)", f"{lstm_mse:.4f}")
m2.metric("LSTM MAPE", f"{lstm_mape:.2f}%")
m3.metric("决策树 MSE", f"{dt_mse:.4f}")
m4.metric("决策树 MAPE", f"{dt_mape:.2f}%")

best_model_name = "LSTM" if lstm_mape < dt_mape else "Decision Tree"
st.markdown(f"""
** 结果讨论 (Discussion):**
通过对比2024年以来的回测测试集，使用 MSE 和 MAPE 作为性能量化指标。
MAPE 衡量预测偏离真实值的平均百分比，更具直观性。
根据当前计算，**{best_model_name}** 具有更低的 MAPE 值，表现更好。
""")

st.markdown("###  下一交易日投资建议")

last_seq_scaled = scaled_data[-SEQ_LEN:]
last_close = df['close'].iloc[-1]

next_lstm_scaled = lstm_model.predict(last_seq_scaled.reshape(1, SEQ_LEN, 1), verbose=0)
next_lstm_real = scaler.inverse_transform(next_lstm_scaled)[0][0]

next_dt_input = last_seq_scaled[-n_features:].reshape(1, -1)
next_dt_scaled = dt_model.predict(next_dt_input)
next_dt_real = scaler.inverse_transform(next_dt_scaled.reshape(-1, 1))[0][0]

best_pred = next_lstm_real if best_model_name == "LSTM" else next_dt_real
expected_change = (best_pred - last_close) / last_close * 100

if expected_change > 1.0:
    action = "🟢 买入 (Buy)"
    insight = "模型预测下一交易日有较明显上涨趋势（预计涨幅 > 1%）。"
elif expected_change < -1.0:
    action = "🔴 卖出 (Sell)"
    insight = "模型预测下一交易日有下行风险（预计跌幅 > 1%）。"
else:
    action = "🟡 持有 / 观望 (Hold)"
    insight = "模型预测下一交易日价格波动较小，建议暂不进行激进操作。"

st.info(f"""
**最新实际收盘价:** {last_close:.2f} 元  
**LSTM 预测明日收盘价:** {next_lstm_real:.2f} 元  
**决策树 预测明日收盘价:** {next_dt_real:.2f} 元  

**综合推荐策略:** 基于最优模型预测，预期明日价格变化约为 **{expected_change:.2f}%**。  
**操作建议:** **{action}**  

**分析洞察:** {insight}  
""")
