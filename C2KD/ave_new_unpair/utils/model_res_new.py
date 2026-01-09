from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, modality, num_classes=1000, num_frame=10, pool='avgpool', zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.modality = modality
        self.pool = pool
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if modality == 'audio':
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif modality == 'visual':
            self.conv1 = nn.Conv2d(3 * num_frame, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            raise NotImplementedError('Incorrect modality, should be audio or visual but got {}'.format(modality))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        if self.pool == 'avgpool':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 8192

        # if modality == 'audio':
        #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)  # 8192
        # elif modality == 'visual':
        #     self.avgpool = nn.AdaptiveAvgPool3d(1)
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward_encoder(self, x):
        if self.modality == 'visual':
            (B, C, T, H, W) = x.size()
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(B, C * T, H, W)
        else:
            x = x.unsqueeze(1)
        x = x.float()

        # --- Backbone ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x # (Optional: Nếu cần intermediate feature thì return thêm)
        x = self.maxpool(x)

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x
        
        # --- Pooling & Flatten ---
        x_512 = self.avgpool(x)
        feature_vector = x_512.reshape(x_512.shape[0], -1) #  phi
        
        return feature_vector, [f0, f1, f2, f3, f4]

    def forward_head(self, feature_vector):
        logits = self.fc(feature_vector)
        return logits
    
    def forward(self, x):
        feature, feature_maps = self.forward_encoder(x)

        logits = self.forward_head(feature)

        return logits, feature, feature_maps

    
    # def forward(self, x):
    #     if self.modality == 'visual':
    #         (B, C, T, H, W) = x.size()
    #         x = x.permute(0, 2, 1, 3, 4).contiguous()
    #         x = x.view(B, C * T, H, W)
    #     else:
    #         x = x.unsqueeze(1)
    #     # x = x.unsqueeze(1)
    #     x = x.float()
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     f0 = x
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     f1 = x
    #     x = self.layer2(x)
    #     f2 = x
    #     x = self.layer3(x)
    #     f3 = x
    #     x_512 = self.avgpool(self.layer4(x))
    #     x_512 = x_512.reshape(x_512.shape[0], -1)
    #     f4 = x
    #     out = self.fc(x_512)

    #     return out, x_512, [f0, f1, f2, f3, f4]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



def _resnet(arch, block, layers, modality, num_classes, num_frame):
    model = ResNet(block, layers, modality, num_classes=num_classes, num_frame=num_frame)
    return model



# class AudioNet(nn.Module):
#     """AudioNet"""

#     def __init__(self, args):
#         super(AudioNet, self).__init__()
#         self.arch = args.audio_arch
#         if self.arch == 'resnet18':
#             layers = [2, 2, 2, 2]
#         if self.arch == 'resnet50':
#             layers = [3, 4, 6, 3]
#         self.backbone = _resnet('resnet_x', BasicBlock, layers, modality='audio', num_classes=28, num_frame=args.num_frame)

#     def fc(self, x):
#         return self.backbone.fc(x)

#     def forward_encoder(self, x):
#         return self.backbone.forward_encoder(x)
    
#     def forward_head(self, feature_vector):
#         return self.backbone.forward_head(feature_vector)

#     def forward(self, x):
#         return self.backbone(x)

class FCReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_C1, s_C2, use_relu=True):
        super(FCReg, self).__init__()
        self.use_relu = use_relu
        self.fc = nn.Linear(s_C1, s_C2)
        self.bn = nn.BatchNorm1d(s_C2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)
        return x

# class ImageNet(nn.Module):
#     """ImageNet"""
#     def __init__(self, args):
#         super(ImageNet, self).__init__()
#         self.arch = args.image_arch
#         if self.arch == 'resnet18':
#             layers = [2, 2, 2, 2]
#         if self.arch == 'resnet50':
#             layers = [3, 4, 6, 3]
#         self.backbone = _resnet('resnet_x', BasicBlock, layers, modality='visual', num_classes=28, num_frame=args.num_frame)

#     def fc(self, x):
#         return self.backbone.fc(x)
#     def forward_encoder(self, x):
#         return self.backbone.forward_encoder(x)
#     def forward_head(self, feature_vector):
#         return self.backbone.forward_head(feature_vector)
#     def forward(self, x):
#         return self.backbone(x)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# DropPath (Stochastic Depth)
# -------------------------
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# -------------------------
# ViT building blocks
# -------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path_prob=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Conv2d patch embedding: (B,C,H,W) -> (B, N, D)"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B,C,H,W)
        x = self.proj(x)  # (B, D, H', W')
        Hp, Wp = x.shape[-2], x.shape[-1]
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x_flat, (Hp, Wp), x  # return also (B,D,H',W') for f0


def interpolate_pos_encoding(pos_embed, Hp, Wp):
    """
    pos_embed: (1, 1+N, D) where N = H0*W0 at init.
    Return: (1, 1+Hp*Wp, D)
    """
    N = pos_embed.shape[1] - 1
    D = pos_embed.shape[2]
    if N == Hp * Wp:
        return pos_embed

    cls_pos = pos_embed[:, :1, :]
    patch_pos = pos_embed[:, 1:, :]  # (1, N, D)

    h0 = w0 = int(math.sqrt(N))
    patch_pos = patch_pos.reshape(1, h0, w0, D).permute(0, 3, 1, 2)  # (1,D,h0,w0)
    patch_pos = F.interpolate(patch_pos, size=(Hp, Wp), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, Hp * Wp, D)  # (1,Hp*Wp,D)

    return torch.cat((cls_pos, patch_pos), dim=1)


class VisionTransformerBackbone(nn.Module):
    """
    Drop-in backbone giống ResNet của bạn:
      forward_encoder -> (feature_vector, [f0..f4])
      forward_head    -> logits
    """
    def __init__(self, modality, num_classes=1000, num_frame=10,
                 img_size=224, patch_size=16, in_chans=None,
                 embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path_rate=0.1,
                 use_cls_token=True):
        super().__init__()
        self.modality = modality
        self.num_frame = num_frame
        self.use_cls_token = use_cls_token

        if in_chans is None:
            if modality == "visual":
                in_chans = 3 * num_frame
            elif modality == "audio":
                in_chans = 1
            else:
                raise NotImplementedError(f"Incorrect modality: {modality}")

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None

        # init pos_embed theo img_size mặc định (có nội suy khi H,W khác)
        num_patches_init = (img_size // patch_size) * (img_size // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches_init, embed_dim))
        self.pos_drop = nn.Dropout(p=drop)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                  drop=drop, attn_drop=attn_drop, drop_path_prob=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.fc = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _tokens_to_map(self, x_tokens, Hp, Wp):
        """
        x_tokens: (B, 1+N, D) if has cls else (B, N, D)
        return feature map: (B, D, Hp, Wp)
        """
        if self.use_cls_token:
            x_tokens = x_tokens[:, 1:, :]
        B, N, D = x_tokens.shape
        x_map = x_tokens.transpose(1, 2).reshape(B, D, Hp, Wp)
        return x_map

    def forward_encoder(self, x):
        # input formatting giống ResNet của bạn
        if self.modality == "visual":
            (B, C, T, H, W) = x.size()
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(B, C * T, H, W)
        else:
            x = x.unsqueeze(1)
        x = x.float()

        # Patch embed
        x_tokens, (Hp, Wp), x_patchmap = self.patch_embed(x)  # x_patchmap: (B,D,Hp,Wp)
        f0 = x_patchmap

        # Add cls + pos
        if self.use_cls_token:
            cls = self.cls_token.expand(x_tokens.size(0), -1, -1)
            x_tokens = torch.cat((cls, x_tokens), dim=1)  # (B, 1+N, D)

        pos = interpolate_pos_encoding(self.pos_embed, Hp, Wp)
        x_tokens = x_tokens + pos
        x_tokens = self.pos_drop(x_tokens)

        # pick 4 intermediate depths để tạo f1..f4
        depth = len(self.blocks)
        pick = [
            max(depth // 4 - 1, 0),
            max(depth // 2 - 1, 0),
            max((3 * depth) // 4 - 1, 0),
            depth - 1
        ]
        feats = {}

        for i, blk in enumerate(self.blocks):
            x_tokens = blk(x_tokens)
            if i in pick:
                feats[i] = self._tokens_to_map(x_tokens, Hp, Wp)

        x_tokens = self.norm(x_tokens)

        # feature vector
        if self.use_cls_token:
            feature_vector = x_tokens[:, 0]  # CLS
        else:
            feature_vector = x_tokens.mean(dim=1)  # mean pool tokens

        # feature maps list theo format [f0..f4]
        f1 = feats[pick[0]]
        f2 = feats[pick[1]]
        f3 = feats[pick[2]]
        f4 = feats[pick[3]]

        return feature_vector, [f0, f1, f2, f3, f4]

    def forward_head(self, feature_vector):
        return self.fc(feature_vector)

    def forward(self, x):
        feature, feature_maps = self.forward_encoder(x)
        logits = self.forward_head(feature)
        return logits, feature, feature_maps


def _vit(arch, modality, num_classes, num_frame, img_size=224):
    # preset configs (bạn có thể đổi embed_dim/depth/head tùy ý)
    arch = arch.lower()
    if arch in ["vit_tiny", "vit_tiny16"]:
        cfg = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    elif arch in ["vit_small", "vit_small16"]:
        cfg = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    elif arch in ["vit_base", "vit_base16"]:
        cfg = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    else:
        raise ValueError(f"Unknown vit arch: {arch}")

    model = VisionTransformerBackbone(
        modality=modality,
        num_classes=num_classes,
        num_frame=num_frame,
        img_size=img_size,
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.1,
        use_cls_token=True
    )
    return model

class ImageNet(nn.Module):
    """ImageNet"""
    def __init__(self, args):
        super(ImageNet, self).__init__()
        self.arch = args.image_arch.lower()

        if self.arch.startswith("vit"):
            # Bạn có thể thêm args.image_size nếu muốn; không có thì mặc định 224
            img_size = getattr(args, "image_size", 224)
            self.backbone = _vit(self.arch, modality="visual", num_classes=28,
                                 num_frame=args.num_frame, img_size=img_size)
        else:
            if self.arch == 'resnet18':
                layers = [2, 2, 2, 2]
            elif self.arch == 'resnet50':
                layers = [3, 4, 6, 3]
            else:
                raise ValueError(f"Unknown image_arch: {args.image_arch}")

            self.backbone = _resnet('resnet_x', BasicBlock, layers,
                                   modality='visual', num_classes=28, num_frame=args.num_frame)

    def fc(self, x):
        return self.backbone.fc(x)

    def forward_encoder(self, x):
        return self.backbone.forward_encoder(x)

    def forward_head(self, feature_vector):
        return self.backbone.forward_head(feature_vector)

    def forward(self, x):
        return self.backbone(x)

class AudioNet(nn.Module):
    """AudioNet"""
    def __init__(self, args):
        super(AudioNet, self).__init__()
        self.arch = args.audio_arch.lower()

        if self.arch.startswith("vit"):
            img_size = getattr(args, "audio_size", 224)  # hoặc kích thước spec của bạn
            self.backbone = _vit(self.arch, modality="audio", num_classes=28,
                                 num_frame=args.num_frame, img_size=img_size)
        else:
            if self.arch == 'resnet18':
                layers = [2, 2, 2, 2]
            elif self.arch == 'resnet50':
                layers = [3, 4, 6, 3]
            else:
                raise ValueError(f"Unknown audio_arch: {args.audio_arch}")

            self.backbone = _resnet('resnet_x', BasicBlock, layers,
                                    modality='audio', num_classes=28, num_frame=args.num_frame)

    def fc(self, x):
        return self.backbone.fc(x)

    def forward_encoder(self, x):
        return self.backbone.forward_encoder(x)

    def forward_head(self, feature_vector):
        return self.backbone.forward_head(feature_vector)

    def forward(self, x):
        return self.backbone(x)

