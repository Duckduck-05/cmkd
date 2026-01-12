import torch
import torch.nn as nn
import torch.nn.functional as F
import timm # Yêu cầu: pip install timm

# =============================================================================
# GIỮ NGUYÊN PHẦN RESNET CŨ CỦA BẠN (Để đảm bảo tính tương thích)
# =============================================================================

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
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
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, modality, num_classes=1000, num_frame=10, pool='avgpool', 
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.modality = modality
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        
        # Input layer handling
        if modality == 'audio':
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        elif modality == 'visual':
            self.conv1 = nn.Conv2d(3 * num_frame, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_encoder(self, x):
        if self.modality == 'visual':
            (B, C, T, H, W) = x.size()
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(B, C * T, H, W)
        else:
            x = x.unsqueeze(1)
        
        x = x.float()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x
        x = self.maxpool(x)

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x
        
        x_512 = self.avgpool(x)
        feature_vector = x_512.reshape(x_512.shape[0], -1)
        
        return feature_vector, [f0, f1, f2, f3, f4]

    def forward_head(self, feature_vector):
        return self.fc(feature_vector)
    
    def forward(self, x):
        feature, feature_maps = self.forward_encoder(x)
        logits = self.forward_head(feature)
        return logits, feature, feature_maps

def _resnet(arch, block, layers, modality, num_classes, num_frame):
    return ResNet(block, layers, modality, num_classes=num_classes, num_frame=num_frame)

# =============================================================================
# PHẦN MỚI: TIMM ViT BACKBONE (Thay thế code thủ công)
# =============================================================================

class TimmViT(nn.Module):
    def __init__(self, arch, modality, num_classes, num_frame, pretrained=True):
        super(TimmViT, self).__init__()
        self.modality = modality
        self.num_frame = num_frame
        
        # Xác định số channels đầu vào
        if modality == 'visual':
            in_chans = 3 * num_frame
            img_size = 224
        elif modality == 'audio':
            in_chans = 1
            img_size = (257, 224)
        else:
            raise ValueError(f"Modality {modality} not supported")

        # Load model từ timm
        # global_pool='' để lấy raw tokens, ta sẽ tự xử lý
        self.model = timm.create_model(
            arch, 
            pretrained=pretrained, 
            num_classes=num_classes, 
            in_chans=in_chans,
            global_pool='token', # Dùng CLS token làm feature vector mặc định
            img_size=img_size,      # Quan trọng: Khai báo size thật
            dynamic_img_size=True,  # Quan trọng: Cho phép resize input khác 224
            dynamic_img_pad=True
        )
        
        # Timm tự động xử lý việc sửa Conv2d đầu tiên nếu channels không khớp 3 (bằng cách average hoặc repeat weight)
        # Nên ta không cần code thủ công phần patch_embed.proj

    def forward_encoder(self, x):
        # 1. Pre-process Input (giống logic cũ)
        if self.modality == 'visual':
            (B, C, T, H, W) = x.size()
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(B, C * T, H, W) # (B, 3*T, H, W)
        else:
            x = x.unsqueeze(1) # Audio: (B, 1, H, W)
        
        x = x.float()

        # 2. Extract Features
        # f0: Patch Embeddings (Output của lớp PatchEmbed)
        x_patch = self.model.patch_embed(x) 
        # # Cần định hình lại f0 thành 2D map: (B, N, D) -> (B, D, H, W)
        # B, N, D = x_patch.shape
        # H_grid = W_grid = int(N**0.5) # Giả sử ảnh vuông
        # # f0 = x_patch.transpose(1, 2).reshape(B, D, H_grid, W_grid)
        
        # # Nếu model có cls_token, f0 nên lấy từ patch_embed raw
        # f0 = x_patch.transpose(1, 2).reshape(B, D, H_grid, W_grid)

        if x_patch.dim() == 4:
            # Trường hợp output là (B, D, H, W)
            B, D, H_grid, W_grid = x_patch.shape
            # f0 giữ nguyên dạng 4D
            f0 = x_patch
        else:
            # Trường hợp output là (B, N, D)
            B, N, D = x_patch.shape
            # Cần tính H, W lưới để reshape lại f0
            # Lấy patch_size từ config model
            patch_size = self.model.patch_embed.patch_size
            if isinstance(patch_size, tuple): patch_size = patch_size[0]
            
            H_grid = x.shape[2] // patch_size
            W_grid = x.shape[3] // patch_size
            
            # Reshape (B, N, D) -> (B, D, H, W)
            f0 = x_patch.transpose(1, 2).reshape(B, D, H_grid, W_grid)

        # 3. Lấy Intermediate Feature Maps
        # Timm hỗ trợ lấy output của các transformer blocks cuối cùng
        # reshape=True sẽ tự động biến (B, N, D) thành (B, D, H, W)
        # Ta lấy 4 blocks rải rác để mô phỏng f1, f2, f3, f4
        # indices = [-4, -3, -2, -1] # Lấy 4 block cuối, hoặc bạn có thể chỉnh indices khác
        
        # # Lưu ý: get_intermediate_layers trả về list features
        # intermediate_features = self.model.get_intermediate_layers(x, n=indices, reshape=True)
        
        # if len(intermediate_features) < 4:
        #     # Fallback cho model nhỏ (tiny)
        #     f1 = f2 = f3 = f4 = intermediate_features[-1]
        # else:
        #     f1, f2, f3, f4 = intermediate_features

        # 4. Feature Vector (CLS Token hoặc Global Pool)
        feature_vector = self.model.forward_features(x) 
        # forward_features trả về (B, N, D), ta cần pool hoặc lấy CLS
        if self.model.global_pool == 'token':
            feature_vector = feature_vector[:, 0]
        else:
            feature_vector = feature_vector.mean(dim=1)

        # Trả về format đúng yêu cầu
        return feature_vector, []

    def forward_head(self, feature_vector):
        return self.model.head(feature_vector)

    def forward(self, x):
        feature, feature_maps = self.forward_encoder(x)
        logits = self.forward_head(feature)
        return logits, feature, feature_maps


# =============================================================================
# WRAPPER CLASSES (IMAGE NET & AUDIO NET)
# =============================================================================

class ImageNet(nn.Module):
    """ImageNet Wrapper"""
    def __init__(self, args):
        super(ImageNet, self).__init__()
        self.arch = args.image_arch.lower()
        
        if 'vit' in self.arch:
            # Map tên arch của bạn sang tên của timm (ví dụ)
            timm_arch = 'vit_base_patch16_224' # Mặc định
            if 'tiny' in self.arch: timm_arch = 'vit_tiny_patch16_224'
            elif 'small' in self.arch: timm_arch = 'vit_small_patch16_224'
            
            self.backbone = TimmViT(timm_arch, modality='visual', num_classes=28, num_frame=args.num_frame)
        else:
            # ResNet logic cũ
            if self.arch == 'resnet18': layers = [2, 2, 2, 2]
            elif self.arch == 'resnet50': layers = [3, 4, 6, 3]
            self.backbone = _resnet('resnet_x', BasicBlock, layers, modality='visual', num_classes=28, num_frame=args.num_frame)

    def fc(self, x): return self.backbone.forward_head(x) # Alias cho tương thích code cũ
    def forward_encoder(self, x): return self.backbone.forward_encoder(x)
    def forward_head(self, x): return self.backbone.forward_head(x)
    def forward(self, x): return self.backbone(x)


class AudioNet(nn.Module):
    """AudioNet Wrapper"""
    def __init__(self, args):
        super(AudioNet, self).__init__()
        self.arch = args.audio_arch.lower()

        if 'vit' in self.arch:
            timm_arch = 'vit_base_patch16_224'
            if 'tiny' in self.arch: timm_arch = 'vit_tiny_patch16_224'
            elif 'small' in self.arch: timm_arch = 'vit_small_patch16_224'
            
            self.backbone = TimmViT(timm_arch, modality='audio', num_classes=28, num_frame=args.num_frame)
        else:
            if self.arch == 'resnet18': layers = [2, 2, 2, 2]
            elif self.arch == 'resnet50': layers = [3, 4, 6, 3]
            self.backbone = _resnet('resnet_x', BasicBlock, layers, modality='audio', num_classes=28, num_frame=args.num_frame)

    def fc(self, x): return self.backbone.forward_head(x)
    def forward_encoder(self, x): return self.backbone.forward_encoder(x)
    def forward_head(self, x): return self.backbone.forward_head(x)
    def forward(self, x): return self.backbone(x)