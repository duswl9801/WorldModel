import torch
import torch.nn as nn

INPUT_CHANNEL = 3
EMBEDDING_DIM = 1024

# normalization for encoder
class RMSNorm2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.eps = 1e-8
        # learnable scaling factor for each channel
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.scale


# refer dreamer_v3(couldn't find v1 code) encoder structure for image input. Norm:RMSNormalization, pooling: max pooling
# representation extraction
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(INPUT_CHANNEL, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2) # 64 -> 32
        self.norm1 = RMSNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)  # 32 -> 16
        self.norm2 = RMSNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)  # 16 -> 8
        self.norm3 = RMSNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)  # 8 -> 4
        self.norm4 = RMSNorm2d(256)

        self.gelu = nn.GELU()

        self.flatten = nn.Flatten()
        #self.fc = nn.Linear(256*4*4, EMBEDDING_DIM)

    def forward(self, frame):
        #frame <- preprocessed frame
        x = self.conv1(frame)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.norm2(x)
        x = self.gelu(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.norm3(x)
        x = self.gelu(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.norm4(x)
        x = self.gelu(x)

        x = self.flatten(x)
        # x = self.fc(x)

        return x


def main():
    encoder = Encoder()
    dummy = torch.randn(8, 3, 64, 64)
    out = encoder(dummy)
    print(out.shape) # -> torch.Size([8, 4096])


if __name__ == "__main__":
    main()