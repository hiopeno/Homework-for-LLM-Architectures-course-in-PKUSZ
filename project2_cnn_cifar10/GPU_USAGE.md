# P2 GPU 训练说明

本作业需要训练 `baseline`、`no_bn`、`no_aug` 和 `lr_10x` 四组实验，建议使用课题组服务器提供的 GPU 或 Google Colab GPU。不建议仅使用 CPU 完成全部训练。

## 1. 预计用时

助教参考实现的实测环境为 RTX 5090、PyTorch 2.7.0+cu128，配置为 160 epochs、batch size 128、workers 8。

| 实验 | RTX 5090 实测时间 |
|---|---:|
| `baseline` | 12.92 分钟 |
| `no_bn` | 11.54 分钟 |
| `no_aug` | 13.07 分钟 |
| `lr_10x` | 12.61 分钟 |
| 合计 | 50.14 分钟 |

以上时间不包含首次配置环境和下载数据集。在 5090 上预留约 1 小时即可完成全部实验。

Colab 的 GPU 型号和当前负载不固定。如分配到 T4，可预留约 40–90 分钟/组、3–6 小时完成四组实验。该数字是保守估计，应以实际运行时间为准。

## 2. 课题组服务器

### 2.1 登录与选卡

通过 SSH 登录服务器后，先检查 GPU：

```bash
nvidia-smi
```

选择空闲或负载较低的一张卡。例如准备使用物理编号 2：

```bash
export CUDA_VISIBLE_DEVICES=2
```

设置后，程序内部仍使用 `cuda` 或 `cuda:0`。请遵守课题组的 GPU 使用规则，不要未经协调同时占用多张卡。

### 2.2 环境与数据

进入解压后的 `project2_cnn_cifar10` 目录，使用服务器已配置的 PyTorch 环境，或建立独立环境：

```bash
conda create -n p2-cifar python=3.10 -y
conda activate p2-cifar
pip install -r requirements.txt
```

首次运行可下载 CIFAR-10：

```bash
python prepare_data.py --root cifar_data
```

也可跳过这一步，训练脚本会通过 `download=True` 自动下载。

### 2.3 后台训练

SSH 断开会终止普通前台进程，建议使用 `tmux`：

```bash
tmux new -s p2
```

在 tmux 内顺序执行四组实验：

```bash
python submission/train.py --run baseline --data-root cifar_data --out-dir submission --workers 8 --device cuda
python submission/train.py --run no_bn --data-root cifar_data --out-dir submission --workers 8 --device cuda
python submission/train.py --run no_aug --data-root cifar_data --out-dir submission --workers 8 --device cuda
python submission/train.py --run lr_10x --data-root cifar_data --out-dir submission --workers 8 --device cuda
```

按 `Ctrl-b`，再按 `d` 可退出 tmux 而不停止训练；重新登录后使用：

```bash
tmux attach -t p2
```

共享服务器上建议四组实验依次运行，不要为缩短个人等待时间并行占用多张 GPU。

## 3. Google Colab

### 3.1 开启 GPU

在 Colab 中选择“代码执行程序 → 更改代码执行程序类型 → T4 GPU”，然后检查：

```python
!nvidia-smi
```

### 3.2 挂载 Google Drive

Colab 运行时可能会被回收，建议将作业和训练产物放在 Google Drive：

```python
from google.colab import drive
drive.mount("/content/drive")
```

将 `P2_student_release.zip` 上传到 Drive 后，按实际路径解压：

```python
!unzip -q "/content/drive/MyDrive/P2_student_release.zip" -d "/content/drive/MyDrive/P2"
%cd "/content/drive/MyDrive/P2/project2_cnn_cifar10"
!pip install -r requirements.txt
```

完成代码后，将 `starter` 复制为 `submission`，或者直接将完成的文件放入 `submission/`。

### 3.3 运行实验

```python
!python prepare_data.py --root cifar_data
!python submission/train.py --run baseline --data-root cifar_data --out-dir submission --workers 2 --device cuda
!python submission/train.py --run no_bn --data-root cifar_data --out-dir submission --workers 2 --device cuda
!python submission/train.py --run no_aug --data-root cifar_data --out-dir submission --workers 2 --device cuda
!python submission/train.py --run lr_10x --data-root cifar_data --out-dir submission --workers 2 --device cuda
```

每组实验结束后立即检查日志和输出文件。Colab 若分配不到 GPU、运行时被回收或达到使用限制，请改用课题组服务器，不要等到截止日才首次训练。

## 4. 训练后检查

至少确认：

```text
submission/model.py
submission/train.py
submission/weights.pt
submission/results.csv
submission/logs/baseline.csv
submission/logs/no_bn.csv
submission/logs/no_aug.csv
submission/logs/lr_10x.csv
```

然后运行公开测试：

```bash
python public_test.py --submission submission
```

如出现 CUDA 内存不足，先检查是否有其他进程占用 GPU；必要时可减小 `--batch-size`，但四组实验必须使用相同 batch size。
