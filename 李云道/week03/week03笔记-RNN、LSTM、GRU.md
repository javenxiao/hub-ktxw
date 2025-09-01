# 循环神经网络（RNN）、LSTM与GRU

## 1. 传统RNN（Recurrent Neural Network）

### 基本结构

传统RNN通过循环结构处理序列数据，能够捕捉时间维度上的依赖关系。

### 计算过程

- **输入**：当前时间步的输入 xt和上一时间步的隐藏状态 ht−1

- **计算**：
  $$
  h_t=tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
  $$
  
- **输出**：
  $$
  y_t=W_{hy}h_t+b_y
  $$
  

### 缺点

- 梯度消失/爆炸问题，循环带来的
- 难以学习长期依赖关系，早期数据关系传播中逐渐弱化

### torch示例
```python
import torch
import torch.nn as nn
rnn = nn.RNN(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
```
------

## 2. LSTM（Long Short-Term Memory）

### 设计思想

通过引入门控机制和细胞状态，选择性记住和忘记信息，解决长期依赖问题。

### 核心组件

- **细胞状态（Ct）**：信息传递的主干线
- **三个门控**：
  - 遗忘门（Forget Gate）
  - 输入门（Input Gate）
  - 输出门（Output Gate）

### 计算过程（时间步t）

1. **遗忘门**：决定从细胞状态中丢弃哪些信息
   $$
   f_t = \sigma(W_f · [h_{t-1},x_t]+b_f)
   $$
   
2. **输入门**：决定哪些新信息存入细胞状态
   $$
   i_t= \sigma(W_i · [h_{t-1},x_t]+b_i)
   $$

   $$
   \tilde{C}_t=tanh(W_C⋅[h_{t−1},x_t]+b_C)
   $$
   
3. **更新细胞状态**：
$$
C_t=f_t⊙C_{t−1}+i_t⊙\tilde{C}_t
$$

4. **输出门**：决定输出哪些信息

$$
o_t=σ(W_o⋅[h_{t−1},x_t]+b_o)
$$

$$
   h_t=o_t⊙tanh(C_t)
$$

### torch示例
```python
import torch
import torch.nn as nn
rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
```
------

## 3. GRU（Gated Recurrent Unit）

### 设计思想

LSTM的简化版本，合并细胞状态和隐藏状态，减少参数数量但保持类似性能。

### 核心组件

- **两个门控**：
  - 更新门（Update Gate）
  - 重置门（Reset Gate）

### 计算过程（时间步t）

1. **更新门**：控制前一状态信息的保留程度
$$
z_t=σ(W_z⋅[h_{t−1},x_t]+b_z)
$$
2. **重置门**：控制前一状态信息的忽略程度
$$
r_t=σ(W_r⋅[h_{t−1},x_t]+b_r)
$$
3. **候选隐藏状态**：
$$
\tilde{h}_t=tanh(W⋅[r_t⊙h_{t−1},x_t]+b)
$$
4. **最终隐藏状态**：
$$
h_t=(1−z_t)⊙h_{t−1}+z_t⊙\tilde{h}t
$$

### torch示例
```python
import torch
import torch.nn as nn
rnn = nn.GRU(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
```
------

## 4. 对比总结

| 特性       | RNN  | LSTM  | GRU   |
| ---------- | ---- | ----- | ----- |
| 门控机制   | 无   | 3个门 | 2个门 |
| 参数数量   | 少   | 多    | 中等  |
| 计算复杂度 | 低   | 高    | 中等  |
| 细胞状态   | 无   | 有    | 无    |
| 长期依赖   | 差   | 优秀  | 良好  |
| 训练速度   | 快   | 慢    | 中等  |

## 5. 使用

- **简单序列任务**：可选择传统RNN或GRU
- **长序列依赖**：优先选择LSTM
- **资源受限环境**：考虑使用GRU平衡性能与效率

------

