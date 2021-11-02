'''
In AI applications that are safety-critical (e.g., medical decision making and autonomous driving) or where the data is inherently noisy (e.g., natural language understanding), it is important for a deep classifier to reliably quantify its uncertainty. The deep classifier should be able to be aware of its own limitations and when it should hand control over to the human experts. This tutorial shows how to improve a deep classifier's ability in quantifying uncertainty using a technique called Spectral-normalized Neural Gaussian Process (SNGP).

The core idea of SNGP is to improve a deep classifier's distance awareness by applying simple modifications to the network. A model's distance awareness is a measure of how its predictive probability reflects the distance between the test example and the training data. This is a desirable property that is common for gold-standard probablistic models (e.g., the Gaussian process with RBF kernels) but is lacking in models with deep neural networks. SNGP provides a simple way to inject this Gaussian-process behavior into a deep classifier while maintaining its predictive accuracy.

This tutorial implements a deep residual network (ResNet)-based SNGP model on the two moons dataset, and compares its uncertainty surface with that of two other popular uncertainty approaches - Monte Carlo dropout and Deep ensemble).

This tutorial illustrates the SNGP model on a toy 2D dataset. For an example of applying SNGP to a real-world natural language understanding task using BERT-base, please see the SNGP-BERT tutorial. For high-quality implementations of SNGP model (and many other uncertainty methods) on a wide variety of benchmark datasets (e.g., CIFAR-100, ImageNet, Jigsaw toxicity detection, etc), please check out the Uncertainty Baselines benchmark.

'''

'''
About SNGP
Spectral-normalized Neural Gaussian Process (SNGP) is a simple approach to improve a deep classifier's uncertainty quality while maintaining a similar level of accuracy and latency. Given a deep residual network, SNGP makes two simple changes to the model:

It applies spectral normalization to the hidden residual layers.
It replaces the Dense output layer with a Gaussian process layer.
'''

'''
![](http://tensorflow.org/tutorials/understanding/images/sngp.png?hl=zh_cn)
'''

'''
Compared to other uncertainty approaches (e.g., Monte Carlo dropout or Deep ensemble), SNGP has several advantages:

It works for a wide range of state-of-the-art residual-based architectures (e.g., (Wide) ResNet, DenseNet, BERT, etc).
It is a single-model method (i.e., does not rely on ensemble averaging). Therefore SNGP has a similar level of latency as a single deterministic network, and can be scaled easily to large datasets like ImageNet and Jigsaw Toxic Comments classification.
It has strong out-of-domain detection performance due to the distance-awareness property.
The downsides of this method are:

The predictive uncertainty of a SNGP is computed using the Laplace approximation. Therefore theoretically, the posterior uncertainty of SNGP is different from that of an exact Gaussian process.

SNGP training needs a covariance reset step at the begining of a new epoch. This can add a tiny amount of extra complexity to a training pipeline. This tutorial shows a simple way to implement this using Keras callbacks.
'''

'''
在对安全至关重要（例如，医疗决策和自动驾驶）或数据本身具有噪声（例如，自然语言理解）的 AI 应用中，深度分类器可靠地量化其不确定性非常重要。深度分类器应该能够意识到自己的局限性，以及何时应该将控制权交给人类专家。本教程展示了如何使用称为频谱归一化神经高斯过程 ( SNGP )的技术提高深度分类器量化不确定性的能力。

SNGP 的核心思想是通过对网络进行简单的修改来提高深度分类器的距离意识。模型的距离意识是衡量其预测概率如何反映测试示例与训练数据之间的距离的度量。这是黄金标准概率模型（例如，具有 RBF 核的高斯过程）常见的理想特性，但在具有深度神经网络的模型中却缺乏。SNGP 提供了一种简单的方法来将这种高斯过程行为注入深度分类器，同时保持其预测准确性。

本教程在两颗卫星数据集上实现了一个基于深度残差网络 (ResNet) 的 SNGP 模型，并将其不确定性表面与其他两种流行的不确定性方法——蒙特卡罗辍学和深度集成）的不确定性表面进行了比较。

本教程说明了玩具 2D 数据集上的 SNGP 模型。有关使用 BERT-base 将 SNGP 应用于现实世界自然语言理解任务的示例，请参阅SNGP-BERT 教程。要在各种基准数据集（例如CIFAR-100、ImageNet、Jigsaw 毒性检测等）上高质量地实现 SNGP 模型（以及许多其他不确定性方法），请查看不确定性基线基准。

关于SNGP
频谱归一化神经高斯过程 (SNGP) 是一种简单的方法，可以提高深度分类器的不确定性质量，同时保持相似的准确度和延迟水平。给定一个深度残差网络，SNGP 对模型做了两个简单的改变：

它将频谱归一化应用于隐藏的残差层。
它将密集输出层替换为高斯处理层。
与其他不确定性方法（例如，Monte Carlo dropout 或 Deep ensemble）相比，SNGP 具有以下几个优点：

它适用于各种最先进的基于残差的架构（例如，（Wide）ResNet、DenseNet、BERT 等）。
它是一种单模型方法（即，不依赖于整体平均）。因此，SNGP 具有与单个确定性网络相似的延迟水平，并且可以轻松扩展到大型数据集，如ImageNet和Jigsaw Toxic Comments 分类。
由于距离感知特性，它具有很强的域外检测性能。
这种方法的缺点是：

使用拉普拉斯近似计算 SNGP 的预测不确定性。因此理论上SNGP的后验不确定性不同于精确高斯过程的后验不确定性。

SNGP 训练需要在新纪元开始时进行协方差重置步骤。这可能会给训练管道增加一点额外的复杂性。本教程展示了一种使用 Keras 回调实现此功能的简单方法。

'''

## set up

!pip uninstall -y tf-nightly keras-nightly

!pip install tensorflow
!pip install --use-deprecated=leagcy-resolver tf-models-official

# refresh pkg-resources so it takes the changes into account
import pkg_resources
import importlib
importlib.reload(pkg_resources)

import.rcParams['figure.dpi'] = 1200

DEFAULT_X_RANGE = (-3.5, 3.5)
DEFAULT_Y_RANGE = (-2.5, 2.5)
DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
DEFAULT_NORM = colors.Normalize(vmin=0, vmax=1,)
DEFAULT_N_GRID = 100

# 从两个月亮数据集创建训练和评估数据集
def make_training_data(sample_size=500):
  """Create two moon training dataset."""
	train_examples, train_labels = sklearn.datasets.make_moons(
    	n_samples=2 * sample_size, noise=0.1)

  # Adjust data position slightly.
	train_examples[train_labels == 0] += [-0.1, 0.2]
  	train_examples[train_labels == 1] += [0.1, -0.2]

  	return train_examples, train_labels


























