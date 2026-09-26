# 阶段作业 1：用 NumPy 手写 MLP 识别 MNIST

**发布日期**：2026 年 9 月 23 日（周三）  
**截止时间**：2026 年 10 月 7 日（周三）24:00（北京时间，即 10 月 8 日 00:00）  
**运行环境**：Python ≥ 3.10、NumPy  
**训练时间限制**：普通笔记本 CPU 上不超过 10 分钟

## 1. 作业目标

只使用 NumPy 和 Python 标准库，实现一个用于 MNIST 手写数字分类的多层感知机。你需要完成前向传播、反向传播、Softmax 交叉熵、mini-batch SGD、权重保存与加载。

禁止导入 `torch`、`tensorflow`、`jax`、`sklearn`、`scipy` 等第三方库；发现违规 import，本次作业记 0 分。

## 2. 数据

发布包中的 `mnist.npz` 已完成展平、类型转换和 `[0,1]` 归一化，可直接读取：

```python
d = np.load("mnist.npz")
x_train, y_train = d["x_train"], d["y_train"]
x_test, y_test = d["x_test"], d["y_test"]
```

数组格式：

- `x_train`: `(60000, 784)`，`float32`
- `y_train`: `(60000,)`，`int64`
- `x_test`: `(10000, 784)`，`float32`
- `y_test`: `(10000,)`，`int64`

公开测试集用于开发和验证；正式性能分在助教持有的隐藏扰动集上计算。

## 3. 需要完成的文件

将 `starter/` 复制或改名为 `submission/`，完成其中所有 TODO：

```text
submission/
  layers.py
  mlp.py
  train.py
```

### `layers.py`

必须保持以下接口和属性名：

```python
class Linear:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        self.W
        self.b
    def forward(self, x):
        ...
    def backward(self, grad_out):
        # 保存 self.dW、self.db，并返回 dL/dx
        ...

class ReLU:
    def forward(self, x):
        ...
    def backward(self, grad_out):
        ...

class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        # 返回 batch 平均损失
        ...
    def backward(self):
        # 返回已经除以 batch size 的 dL/dlogits
        ...
```

评测约定每次都先调用 `forward`，再调用对应的 `backward`。

### `mlp.py`

```python
def build_model(seed: int = 0) -> "MLP":
    ...

class MLP:
    def predict(self, X):
        ...
    def save(self, path):
        ...
    def load(self, path):
        ...
```

网络输入维度必须为 784，输出维度必须为 10；隐藏层结构可以自行设计。

### `train.py`

运行命令：

```bash
python submission/train.py --data mnist.npz \
  --out submission/weights.npz \
  --log submission/train_log.csv
```

## 4. 公开自测

完成代码后，在发布包根目录运行：

```bash
python public_test.py --submission submission
```

公开测试通过不代表获得满分；正式评分还包括更多梯度检查、隐藏集准确率和日志一致性检查。

## 5. 最终提交内容

压缩包命名为 `P1_学号_姓名.zip`，内部必须包含：

```text
submission/
  layers.py
  mlp.py
  train.py
  weights.npz
  train_log.csv
```

`train_log.csv` 至少 5 行，列名必须为：

```csv
epoch,train_loss,val_acc
```

`val_acc` 使用 `[0,1]` 小数，例如 `0.973`，不要写成 `97.3`。

## 6. 评分标准

| 项目 | 分值 | 说明 |
|---|---:|---|
| import 合规 | 门槛 | 违规直接记 0 分 |
| 前向正确性 | 10 | Linear、ReLU、SoftmaxCE |
| 梯度检查 | 35 | Linear 的 dx/dW/db、ReLU、SoftmaxCE |
| 隐藏集准确率 | 45 | ≥97%: 45；≥95%: 38；≥93%: 32；≥90%: 23；≥80%: 11 |
| 日志一致性 | 10 | 格式、损失趋势及验证准确率一致性 |

## 7. 学术诚信

- 课程将剔除模板代码后进行代码查重。
- 助教会随机抽取提交，按其代码重新训练并验证权重与日志。
- 日志和权重必须来自同一次真实实验。
- 提交时间以问卷平台记录的成功提交时间为准；截止前重复提交时，以最后一次有效提交为准。
- 迟交不足 24 小时按 1 天计算，每迟交 1 天扣除总分的 20%。计算公式为：`最终成绩 = 自动评分 × max(0, 1 - 20% × 迟交天数)`；迟交 5 天及以上记 0 分。
- 如遇平台故障，应在截止前保留带时间信息的截图，并及时联系助教。
