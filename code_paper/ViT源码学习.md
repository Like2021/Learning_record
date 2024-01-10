## ViT

[官方代码](https://github.com/google-research/vision_transformer/tree/main)

### Patch输入处理

假设输入图像尺寸为**224×224×3**，一个**token**的尺寸为**16×16×3**，把图像分割成**(224/16)×(224/16)**，即**196**个**patch**，然后展平每一个**patch**成**16×16×3**，即**768**维，最后得到输入为**(196, 768)**，对应**[num_token, token_dim]**。

**patch**裁剪的具体实现为卷积：

```python
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, image_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        """
        Map input tensor to patch.
        Args:
            image_size: input image size
            patch_size: patch size
            in_c: number of input channels
            embed_dim: embedding dimension. dimension = patch_size * patch_size * in_c
            norm_layer: The function of normalization
        """
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # The input tensor is divided into patches using 16x16 convolution
        # 步长和卷积核都是patch_size
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]  [B, 768, 14, 14] -> [B, 768, 196]
        # transpose: [B, C, HW] -> [B, HW, C]  [B, 768, 196] -> [B, 196, 768]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x
```

#### Position Embedding

位置信息以***Add***的形式融合到输入tensor中。

#### Class Token

此特征向量维度为[1, 768]，用于后续单独取出来做分类预测，所以须以***Concat***的形式融合到输入tensor中，输入维度变成**(197, 768)**。

### Encoder

再整合好patch输入之后，接一个Transformer常用的Encoder，进一步编码。

### MLP

用一个线性层作为输出头，输出预测结果。