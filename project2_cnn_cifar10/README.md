# 阶段作业 2：CIFAR-10 CNN 与消融实验

**截止时间**：由课程通知确定  
**环境**：Python ≥ 3.10、PyTorch ≥ 2.0、torchvision  
**训练预算**：单张免费 Colab T4 上每个实验不超过 2 小时

GPU 环境、课题组服务器和 Colab 使用方法见 `GPU_USAGE.md`。

## 1. 任务

从随机初始化开始训练一个不超过 5,000,000 参数的 CIFAR-10 卷积网络。禁止使用预训练权重或外部训练数据。

需要完成四次相同预算的实验：

| run_name | 唯一变化 |
|---|---|
| `baseline` | 完整训练配方 |
| `no_bn` | 模型中移除所有 BatchNorm |
| `no_aug` | 关闭训练数据增广 |
| `lr_10x` | baseline 初始学习率乘以 10 |

除表中指定变量外，其余结构、优化器、epoch、batch size 等设置必须保持一致。

## 2. 模型接口

在 `model.py` 中实现：

```python
def build_model(variant: str = "baseline") -> torch.nn.Module:
    ...
```

`variant` 只接受 `baseline` 和 `no_bn`。两者应保持相同主体结构，只改变 BatchNorm。

grader 输入为未经归一化的 `[0,1]`、`float32` 张量，形状 `(N,3,32,32)`。如训练使用均值方差归一化，必须在模型内部完成，例如使用 `register_buffer` 保存均值和标准差。模型输出必须为 `(N,10)` logits。

## 3. 数据与训练

可使用 torchvision 下载官方 CIFAR-10：

```python
torchvision.datasets.CIFAR10(root, train=True, download=True, ...)
torchvision.datasets.CIFAR10(root, train=False, download=True, ...)
```

数据集不放在作业压缩包中。首次使用可先执行：

```bash
python prepare_data.py --root cifar_data
```

也可直接运行训练脚本；其中的 `download=True` 会自动下载官方训练集和测试集。

只允许使用官方 50,000 张训练图像训练。公开测试集用于开发验证，正式性能分在助教隐藏扰动集上计算。

推荐 baseline 使用随机裁剪、随机水平翻转、SGD momentum、weight decay 和学习率调度。消融实验的结果不要求都高，但必须真实记录。

## 4. 公开自测

将 `starter/` 复制或改名为 `submission/`，完成 `model.py` 和 `train.py` 后运行：

```bash
python public_test.py --submission submission
```

该测试检查参数量、输入输出形状和 BatchNorm 开关，不检查最终准确率。

## 5. 提交内容

压缩包命名为 `P2_学号_姓名.zip`，内部必须包含：

```text
submission/
  model.py
  train.py
  weights.pt
  results.csv
  logs/
    baseline.csv
    no_bn.csv
    no_aug.csv
    lr_10x.csv
```

`weights.pt` 必须是 baseline 的 `state_dict`：

```python
torch.save(model.state_dict(), "weights.pt")
```

四组实验可依次运行：

```bash
python submission/train.py --run baseline --data-root cifar_data --out-dir submission
python submission/train.py --run no_bn --data-root cifar_data --out-dir submission
python submission/train.py --run no_aug --data-root cifar_data --out-dir submission
python submission/train.py --run lr_10x --data-root cifar_data --out-dir submission
```

每份日志至少 10 行，格式为：

```csv
epoch,train_loss,val_acc
0,1.9234,0.4210
```

`results.csv` 必须有四行：

```csv
run_name,test_acc,params,train_minutes
baseline,...,...,...
no_bn,...,...,...
no_aug,...,...,...
lr_10x,...,...,...
```

准确率使用 `[0,1]` 小数。`results.csv`、日志、参数量和权重必须来自真实实验并相互一致。

## 6. 全自动评分

| 项目 | 分值 | 说明 |
|---|---:|---|
| 结构与接口 | 20 | 参数量、输出形状、baseline 含 BN、no_bn 不含 BN |
| 隐藏集准确率 | 50 | ≥91%: 50；≥88%: 42；≥85%: 35；≥80%: 25；≥70%: 12 |
| 消融完备与一致性 | 30 | 四份日志、results 一致性和参数量真实性 |

## 7. 学术诚信

- 禁止使用预训练权重、外部数据或伪造日志。
- 课程将对模型代码和训练产物进行一致性检查与抽查重训。
- 代码会在剔除模板后进行查重。
- 迟交与申诉规则以课程统一通知为准。
