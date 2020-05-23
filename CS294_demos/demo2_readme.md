将原始的`lecture2_autoregressive_models_demos.ipynb`9个小部分，对应着其中的9个小demo：
- `lecture2_demos_section1.py` 一维数据的拟合
- `lecture2_demos_section2.py` 处理二维数据的简单自回归模型
- `lecture2_demos_section3.py` 在MNIST图像集上 RNN、MAED、PixelCNN 自回归模型
- `lecture2_demos_section4.py` PixelCNN 以及 HoriVertStackPixelCNN 的 blindspot与感受野的比较，两者采用的mask不一样
- `lecture2_demos_section5.py` Self-Attention(multi-head attention)自回归模型
- `lecture2_demos_section6.py` 自回归模型中不同的编码顺序，对于二维图像而言有：随机、横扫、竖扫等等
- `lecture2_demos_section7.py` 条件自回归模型
- `lecture2_demos_section8.py` Hierarchy (Grayscale PixelCNNs)
- `lecture2_demos_section9.py` Fast Sampling (Parallel PixelCNNs)

其中每个小demo都可以独立运行，`deepul_helper/models.py` 中包含了上述涉及到的主要模型，其中包含有：
- RNN
- MADE(masked auto decoder encoder)
- WaveNet
- PixelCNN
- PixelCNN 的多个改进（attention、parallel等）
  - self-attention
  
demo中并没有涉及模型的训练，都是直接load `pretrained_models/`下训练好的模型。

其他：

为将文件目录添加为当前环境路径，避免使用pycharm调试时，报错找不到相关模块代码
```python
import sys
sys.path.append("../../deepul-master")
sys.path.append("../demos")
```