import torch.nn as nn

# =========================================================
# 1) Models: ResUNet-like Map + Time-conditioned Discriminator
# =========================================================
class CNNMap(nn.Module):
    """
    Residual transport map T(x) = x + Net(x) for RGB images.
    Input/Output: (B, 3, H, W)
    """
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, 1, 1), nn.GELU())            # 64x64
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2, 4, 2, 1), nn.GELU())           # 32x32
        self.enc3 = nn.Sequential(nn.Conv2d(base*2, base*4, 4, 2, 1), nn.GELU())         # 16x16
        self.enc4 = nn.Sequential(nn.Conv2d(base*4, base*4, 4, 2, 1), nn.GELU())         # 8x8

        # Bottleneck
        self.bot = nn.Sequential(nn.Conv2d(base*4, base*4, 3, 1, 1), nn.GELU())

        # Decoder
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(base*4, base*4, 4, 2, 1), nn.GELU())  # 16x16
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(base*4, base*2, 4, 2, 1), nn.GELU())  # 32x32
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(base*2, base, 4, 2, 1), nn.GELU())    # 64x64

        self.out = nn.Conv2d(base, in_ch, 3, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b  = self.bot(e4)

        d1 = self.dec1(b + e4)      # 16x16
        d2 = self.dec2(d1 + e3)     # 32x32
        d3 = self.dec3(d2 + e2)     # 64x64

        disp = self.out(d3 + e1)
        return x + disp


class CNNDiscriminator(nn.Module):
    """
    Patch-critic with global pooling. Input: RGB(3)
    """
    def __init__(self, img_ch=3, base=64):
        super().__init__()
        in_ch = img_ch
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(base*2, base*4, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(base*4, base*4, 3, 1, 1), nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(base*4, 1)

    def forward(self, x):
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.head(h)