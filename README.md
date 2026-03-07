# Stock Price Prediction (LSTM vs Decision Tree)

基于股票代码展示历史走势，并使用 **LSTM** 与 **Decision Tree** 两种模型进行回测预测；历史数据用黑色线、预测用红色线表示。

## 1. 数据探索与预处理 (Discovery & Preprocessing)

- **描述统计**：对收盘价 `close` 等做描述统计，观察数据特征。
- **清洗与预处理**：缺失值处理（前向填充 ffill + 后向填充 bfill）、数据归一化（Min-Max Scaler 将收盘价缩放至 [0,1] 区间）。

## 2. 模型开发 (Model Development)

- **序列设置**：用过去 **30 天** 收盘价作为输入，预测下一天。
- **数据划分**：2015-2023 年数据作为训练集，2024 年至今数据作为测试集。
- **LSTM 模型**：双层 LSTM（每层 8 个隐藏单元），训练 50 轮。
- **Decision Tree 模型**：最大深度 20，使用最后 10 天特征。

## 3. 模型对比与评估 (Comparison & Evaluation)

- 使用 2024 年至今数据作为测试集，计算 **MSE** 和 **MAPE (%)**。
- 比较 LSTM 与 Decision Tree 的指标，按 MAPE 选择更优模型。

## 4. 预测与可视化 (Prediction & Visualization)

- 绘制两个模型的预测图：黑色线为真实数据，红色线为预测数据。
- 基于最优模型给出下一交易日投资建议（买入/卖出/持有）。

## 数据说明

- 数据位于 `stock/` 目录，CSV 命名格式：`sh600000.csv`、`sz302132.csv` 等。
- 支持直接输入股票代码（如 `600000`、`002195`），系统会自动匹配文件。

## 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Streamlit 应用
streamlit run predict.py
```

在页面输入股票代码（如 `600578` 或 `002195`），即可查看走势、模型对比与投资建议。
