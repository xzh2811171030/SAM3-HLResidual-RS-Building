import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedBoundaryAdapter(nn.Module):

    def __init__(self, in_channels: int = 3, feat_channels: int = 256):
        super().__init__()

        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feat_channels, kernel_size=3, padding=1),
        )

        self.alpha_gate = nn.Parameter(torch.zeros(feat_channels, 1, 1))

    def forward(
        self,
        image: torch.Tensor,
        sam_features: torch.Tensor,
    ) -> torch.Tensor:
        edge = self.edge_conv(image)

        edge_pooled = F.adaptive_avg_pool2d(edge, sam_features.shape[-2:])

        gated = edge_pooled * self.alpha_gate

        return sam_features + gated


class EdgeUncertaintyHead(nn.Module):

    def __init__(
        self,
        feat_channels: int = 256,
        mid_channels: int = 128,
        output_size: int = 512,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(feat_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1),
        )

        self.output_size = output_size

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        x = self.conv(fused_features)
        unc_map = torch.sigmoid(x)
        unc_map = F.interpolate(
            unc_map,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return unc_map


class GBG_SAM3_Module(nn.Module):

    def __init__(
        self,
        image_channels: int = 3,
        feat_channels: int = 256,
        output_size: int = 512,
    ):
        super().__init__()

        self.adapter = GatedBoundaryAdapter(
            in_channels=image_channels,
            feat_channels=feat_channels,
        )

        self.eu_head = EdgeUncertaintyHead(
            feat_channels=feat_channels,
            output_size=output_size,
        )

    def forward(
        self,
        image: torch.Tensor,
        sam_features: torch.Tensor,
    ):
        fused_features = self.adapter(image, sam_features)
        unc_map = self.eu_head(fused_features)
        return fused_features, unc_map


if __name__ == "__main__":
    B = 2
    img = torch.randn(B, 3, 512, 512)
    feats = torch.randn(B, 256, 64, 64)

    module = GBG_SAM3_Module()
    fused, unc = module(img, feats)

    print(f"输入图像:      {img.shape}")
    print(f"SAM3 特征:     {feats.shape}")
    print(f"融合特征:      {fused.shape}")
    print(f"不确定性图:    {unc.shape}")
    print(f"\nalpha_gate 初始值 (应全为零): {module.adapter.alpha_gate.abs().sum().item():.6f}")
    print(f"融合特征 == SAM 特征?  {torch.allclose(fused, feats)}")
