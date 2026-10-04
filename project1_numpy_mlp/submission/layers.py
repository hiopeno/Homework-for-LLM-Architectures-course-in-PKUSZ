"""P1 starter — implement the three layers. Only numpy is allowed."""
import numpy as np


class Linear:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He/Xavier-style init is a good choice; keep attribute names W / b.
        self.W = rng.standard_normal((in_dim, out_dim)) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros(out_dim)
        self.dW = None
        self.db = None
        self.x= None

    def forward(self, x):
        # TODO: cache what backward needs, return x @ W + b
        self.x=x
        return x @ self.W + self.b

    def backward(self, grad_out):
        # TODO: set self.dW, self.db and return dL/dx
        self.dW=self.x.T @ grad_out
        N = self.x.shape[0] #获取N
        self.db = np.ones(N) @ grad_out #构建N个1的行向量
        # 以上两行是基于还原数学推导的coding，可以合并为self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T #backward的返回值用于传给前一层继续反向传播

class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_out):
        return grad_out * (self.x > 0) #这里用*，不同于线性层backward的return

class SoftmaxCrossEntropy:
    def __init__(self):
        self.y = None
        self.probs= None

    def forward(self, logits, y):
        # TODO: return mean cross-entropy loss (float). Tip: subtract row max
        # before exp for numerical stability.

        # output=None
        # i=0
        # exp_sum=np.sum(np.exp(logits))
        # for logit in logits:
        #     output[i++]=np.exp(logit)/exp_sum

        # 以上为初次编程的实现，存在以下问题：
        #   output = None 不能进行 output[i] = ...。
        #   i++ 不是 Python 语法，Python 要写 i += 1。
        #   np.sum(np.exp(logits)) 把整个 batch、所有类别一起求和了；Softmax 应该对每个样本单独按类别求和。
        #   没有减去每行最大值，可能发生数值溢出。
        #   目前计算的是概率，而且没有使用 y 计算交叉熵损失。
        #   还需要保存概率和标签，供 backward() 使用。

        normal_logits=logits-np.max(logits,axis=1,keepdims=True)
        exp_logits=np.exp(normal_logits)
        all_probs=exp_logits/np.sum(exp_logits, axis=1, keepdims=True)
        # 这里SoftmaxCrossEntropy是最后一层了，要输出loss服务于向后传播，不能止步于probs

        # loss=probs-y #这是初次编程的实现
        # 前向传播我们只关注正确分类的概率，这里错误地把所有概率都拿来相减了，并且probs-y是梯度，不是loss
        n = logits.shape[0]
        correct_probs = all_probs[np.arange(n), y]
        loss = -np.mean(np.log(correct_probs + 1e-12))
        # 加上1e-12是为了防止概率因为浮点数下溢变成精确的 0

        self.probs=all_probs #这里不能只存正确项的概率，反向传播需要全部项的概率
        self.y=y
        return float(loss)
        # 这里传回去的就是loss=logy^*，不同于中间任何层（他们传的是中间计算值）

    def backward(self):
        # TODO: return dL/dlogits, shape (N, C), already divided by N.
        # (You proved in Lecture 2 that this is (p - y_onehot) / N.)

        # return (self.probs-self.y)/self.probs.shape[0] #这是初次编程的实现
        # 注意向后传播要考虑所有概率，不能只考虑正确项的概率
        N = self.probs.shape[0]
        C = self.probs.shape[1]

        one_hot_y = np.zeros((N, C))
        one_hot_y[np.arange(N), self.y] = 1

        return (self.probs - one_hot_y) / N
    
        #以上为还原数学推导的代码，实际上可以简化为：
        # grad = self.probs.copy() #防止直接修改self.probs
        # grad[np.arange(n), self.y] -= 1
        # return grad / n