from IMDLBenCo.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm



class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        sobel_x = self.sobel_x.to(pred.device).type_as(pred)
        sobel_y = self.sobel_y.to(pred.device).type_as(pred)

        pred_gx = F.conv2d(pred, sobel_x, padding=1)
        pred_gy = F.conv2d(pred, sobel_y, padding=1)

        target_gx = F.conv2d(target.type_as(pred), sobel_x, padding=1)
        target_gy = F.conv2d(target.type_as(pred), sobel_y, padding=1)

        return self.l1(pred_gx, target_gx) + self.l1(pred_gy, target_gy)



class TRIM(nn.Module):
    def __init__(self, in_channels, mid_channels=128, after_relu=False):
        super().__init__()
        self.after_relu = after_relu

        self.feature_transform = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels)
        )

        self.channel_adapter = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )

        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, base_feat, guidance_feat):
        if self.after_relu:
            base_feat = self.relu(base_feat)
            guidance_feat = self.relu(guidance_feat)

        base_shape = base_feat.shape

        q = self.feature_transform(guidance_feat)
        k = self.feature_transform(base_feat)

        q = F.interpolate(q, size=base_shape[2:], mode='bilinear', align_corners=False)

        sim = torch.sigmoid(self.channel_adapter(k * q))

        v = F.interpolate(guidance_feat, size=base_shape[2:], mode='bilinear', align_corners=False)

        return (1 - sim) * base_feat + sim * v



class TokenGuidanceGenerator(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()

        self.token = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        feat = x.flatten(2).permute(0, 2, 1)   # B, HW, C
        token = self.token.expand(B, -1, -1)

        token_out, attn_map = self.attn(token, feat, feat)

        feat = feat + token_out
        feat = self.norm(feat)

        feat = feat.permute(0, 2, 1).view(B, C, H, W)

        return feat, attn_map


class Structural_Aware_Multi_scale_Decoder(nn.Module):
    def __init__(self, encoder_channels=768, decoder_channels=[256, 128, 64, 32]):
        super().__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(encoder_channels, decoder_channels[0], 3, padding=1),
            nn.GELU()
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[0], decoder_channels[1], 3, padding=1),
            nn.GELU()
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[1], decoder_channels[2], 3, padding=1),
            nn.GELU()
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[2], decoder_channels[3], 3, padding=1),
            nn.GELU()
        )

        self.final_conv = nn.Conv2d(decoder_channels[3], 1, 1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.final_conv(x)


@MODELS.register_module()
class MOFNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            features_only=True,
            out_indices=[3],
        )

        self.C = 768

        # ⭐ Token模块
        self.token_block = TokenGuidanceGenerator(self.C)

        self.seg_decoder = Structural_Aware_Multi_scale_Decoder(self.C)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.C, 1)
        )

        self.trim = TRIM(self.C, 128)

        self.spatial_proj = nn.Conv2d(12, self.C, 1)
        self.freq_proj = nn.Conv2d(3, self.C, 1)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # loss
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.edge_loss = EdgeAwareLoss()

        self.edge_weight = 0.1
        self.token_loss_weight = 0.05


    # ---------------- FFT ----------------
    def freq_features(self, x):
        x_fft = torch.fft.fft2(x, norm="ortho")
        mag = torch.abs(x_fft)
        mag = torch.log1p(mag)
        return (mag - mag.mean(dim=[2,3], keepdim=True)) / (mag.std(dim=[2,3], keepdim=True) + 1e-6)


    # ---------------- Wavelet ----------------
    def wavelet_features(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')

        blocks = x.unfold(2, 2, 2).unfold(3, 2, 2)
        a, b, c, d = blocks[...,0,0], blocks[...,0,1], blocks[...,1,0], blocks[...,1,1]

        LL = (a + b + c + d) * 0.25
        LH = (-a - b + c + d) * 0.25
        HL = (-a + b - c + d) * 0.25
        HH = (a - b - c + d) * 0.25

        def up(x): return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return torch.cat([up(LL), up(LH), up(HL), up(HH)], dim=1)


    
    def forward(self, image, mask=None, label=None, edge_mask=None, **kwargs):

        
        features = self.backbone(image)[0]
        features, token_attn = self.token_block(features)

        B, _, HW = token_attn.shape
        H, W = features.shape[2:]
        token_map = token_attn.view(B, 1, H, W)
        wavelet = self.wavelet_features(image)
        wavelet = F.interpolate(wavelet, size=(H, W), mode='bilinear')
        wavelet = self.spatial_proj(wavelet)

        freq = self.freq_features(image)
        freq = F.interpolate(freq, size=(H, W), mode='bilinear')
        freq = self.freq_proj(freq)

        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)

        guidance = alpha * wavelet + beta * freq

        guidance = guidance * (1 + token_map)

        fused = self.trim(features, guidance)

        # ---------------- decoder ----------------
        seg_pred = self.seg_decoder(fused)

        if mask is not None:
            seg_pred = F.interpolate(seg_pred, size=mask.shape[2:], mode='bilinear')

        cls_pred = self.cls_head(features).reshape(-1)

        output = {}

        if mask is not None and label is not None:

            seg_loss = self.seg_loss(seg_pred, mask)
            cls_loss = self.cls_loss(cls_pred, label.float())
            edge_loss = self.edge_loss(seg_pred, mask)

            # ⭐ token supervision
            token_loss = F.binary_cross_entropy_with_logits(
                F.interpolate(token_map, size=mask.shape[2:], mode='bilinear'),
                mask
            )

            loss = seg_loss + cls_loss + self.edge_weight * edge_loss + self.token_loss_weight * token_loss

            output["backward_loss"] = loss
            output["visual_loss"] = {
                "seg": seg_loss,
                "cls": cls_loss,
                "edge": edge_loss,
                "token": token_loss,
                "total": loss
            }

        output["pred_mask"] = torch.sigmoid(seg_pred)
        output["pred_cls_logit"] = cls_pred

        return output