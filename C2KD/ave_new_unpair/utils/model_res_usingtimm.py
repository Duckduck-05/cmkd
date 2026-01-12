import torch
import torch.nn as nn
import timm
import torchaudio.transforms as T
import torch.nn.functional as F

class Projector(nn.Module):
    """
    MLP Projection Head (SimCLR style): Linear -> BN -> ReLU -> Linear
    Dùng để map feature về không gian chung (common embedding space)
    """
    def __init__(self, in_dim, out_dim=128):
        super(Projector, self).__init__()
        # Layer 1: Giữ nguyên dimension hoặc giảm nhẹ (Hidden)
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True)
        )
        # Layer 2: Ép về dimension mục tiêu (ví dụ 128)
        self.layer2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def get_timm_model_name(arch_name):
    """
    Map tên từ args sang tên chuẩn của TIMM.
    Bạn có thể đổi 'tiny' thành 'small' hoặc 'base' ở đây nếu muốn model to hơn.
    """
    arch_name = arch_name.lower()
    if 'resnet18' in arch_name:
        return 'resnet18'
    elif 'resnet50' in arch_name:
        return 'resnet50'
    elif 'convnext' in arch_name:
        return 'convnext_tiny' # Hoặc 'convnext_small'
    elif 'swin' in arch_name:
        return 'swin_tiny_patch4_window7_224'
    else:
        raise ValueError(f"Architecture {arch_name} is not supported yet.")


class ImageNet(nn.Module):
    """
    Model xử lý Video/Image.
    Input: (B, T, C, H, W) -> Output: Logits, Features
    """
    def __init__(self, args, pretrained=False):
        super(ImageNet, self).__init__()
        self.args = args
        self.pretrained = pretrained

        self.dataset = "AVE"

        if  self.dataset == 'VGGSound':
            n_classes = 309
        elif  self.dataset == 'KineticSound':
            n_classes = 31
        elif  self.dataset == 'CREMAD':
            n_classes = 6
        elif  self.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(self.dataset))
        
        # 1. Xác định tên model TIMM
        timm_model_name = get_timm_model_name(args.image_arch)
        print(f"=> Creating Video Backbone: {timm_model_name} | Pretrained: {self.pretrained}")

        # 2. Khởi tạo Backbone qua TIMM
        # num_classes=0: Chỉ lấy Feature Vector
        # global_pool='avg': Tự động Pooling bất chấp kích thước ảnh đầu vào
        self.backbone = timm.create_model(
            timm_model_name, 
            pretrained=self.pretrained, 
            num_classes=0, 
            global_pool='avg'
        )
        
        # Lấy kích thước feature vector (512, 768, 2048...) tự động
        self.feature_dim = self.backbone.num_features

        # 3. Projector / Classification Head
        self.head_video = nn.Linear(self.feature_dim, n_classes)

        self.projector = Projector(in_dim=self.feature_dim, out_dim=128)

    def forward(self, x):
        # Input x: (B, T, C, H, W)
        
        # Check và permute nếu input bị ngược (B, C, T, H, W)
        if x.dim() == 5 and x.size(1) == 3: 
            x = x.permute(0, 2, 1, 3, 4)
            
        B, T, C, H, W = x.shape
        
        # Gộp Batch và Time để đưa vào Backbone 2D: (B*T, C, H, W)
        x = x.reshape(B * T, C, H, W)

        # Forward qua Backbone -> Ra luôn vector (B*T, Feature_Dim)
        features = self.backbone(x)

        # Tách lại Batch và Time: (B, T, Feature_Dim)
        features = features.view(B, T, -1)
        
        # Temporal Pooling: Trung bình cộng các frame
        final_features = torch.mean(features, dim=1) # (B, Feature_Dim)

        proj_features = self.projector(final_features)

        # Classification
        logits = self.head_video(final_features)
        
        return logits, proj_features, []

    def forward_encoder(self, x):
        # Chỉ lấy feature, không qua logit head (dùng cho alignment loss)
        if x.dim() == 5 and x.size(1) == 3: 
            x = x.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        
        features = self.backbone(x) # (B*T, Dim)
        features = features.view(B, T, -1)
        final_features = torch.mean(features, dim=1) # (B, Dim)
        
        return final_features, []
    
    def forward_head(self, feature_vector):
        return self.head_video(feature_vector)
    
    def fc(self, feature_vector):
        return self.head_video(feature_vector)


class AudioNet(nn.Module):
    """
    Model xử lý Audio (Spectrogram).
    Input: (B, 1, F, T) -> Output: Logits, Features
    """
    def __init__(self, args, pretrained=False):
        super(AudioNet, self).__init__()
        self.args = args
        self.pretrained = pretrained

        self.dataset = "AVE"

        if  self.dataset == 'VGGSound':
            n_classes = 309
        elif  self.dataset == 'KineticSound':
            n_classes = 31
        elif  self.dataset == 'CREMAD':
            n_classes = 6
        elif  self.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(self.dataset))

        # 1. Xác định tên model TIMM
        timm_model_name = get_timm_model_name(args.audio_arch)
        print(f"=> Creating Audio Backbone: {timm_model_name} | Pretrained: {self.pretrained}")

        # 2. Khởi tạo Backbone qua TIMM
        # in_chans=1: Quan trọng để xử lý Spectrogram (ảnh đen trắng)
        self.backbone = timm.create_model(
            timm_model_name, 
            pretrained=self.pretrained, 
            in_chans=1, 
            num_classes=0, 
            global_pool='avg'
        )
        
        self.feature_dim = self.backbone.num_features

        # 3. Projector / Classification Head
        self.head_audio = nn.Linear(self.feature_dim, n_classes)

        self.projector = Projector(in_dim=self.feature_dim, out_dim=128)

        self.mel_scale = T.MelScale(n_mels=128, sample_rate=16000, n_stft=257)

    def forward(self, x):
        # Input x: (B, 1, F, T) - Coi như ảnh 1 channel
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Forward qua Backbone -> Ra luôn vector (B, Feature_Dim)
        x = self.mel_scale(x)

        if 'swin' in self.args.audio_arch:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            # Nếu là ConvNeXt/ResNet -> Resize về (128, 512) cho dài (giữ thông tin thời gian)
            # Hoặc bạn có thể ép về 224x224 luôn cho nhẹ cũng được, tùy bạn chọn
            # x = F.interpolate(x, size=(128, 512), mode='bilinear', align_corners=False)
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        features = self.backbone(x)

        # Classification
        logits = self.head_audio(features)

        proj_features = self.projector(features)
        
        return logits, proj_features, []

    def forward_encoder(self, x):
        return self.backbone(x), []

    def forward_head(self, feature_vector):
        return self.head_audio(feature_vector)
    
    def fc(self, feature_vector):
        return self.head_audio(feature_vector)

