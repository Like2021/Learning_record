## BEVFormer

还未精读代码，感觉BEVFormer的query采样有点类似DeformAttn。

### Enocder

输入是当前帧的**image features**和历史**BEV features**。

#### Temporal Self-Attention

作用：对历史**BEV features**和**BEV Queries**进行时间自注意力机制。



