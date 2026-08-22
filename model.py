import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Z_DIM, NUM_CLASSES, EMBED_DIM


class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM):
        super().__init__()

        self.l_emb = nn.Embedding(num_classes, embed_dim)

        self.fc = nn.Linear(z_dim + embed_dim, 256 * 4 * 4)

        self.dconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, z, labels):
        l_emb = self.l_emb(labels)
        x = torch.concat([z, l_emb], dim=1)

        x = F.relu(self.fc(x))
        x = x.view(-1, 256, 4, 4)

        x = F.relu(self.bn1(self.dconv1(x)))
        x = F.relu(self.bn2(self.dconv2(x)))
        x = F.relu(self.bn3(self.dconv3(x)))
        x = torch.tanh(self.dconv4(x))

        return x


class Discriminator(nn.Module):
    def __init__(self, num_classes = NUM_CLASSES, embed_dim = EMBED_DIM):
        super().__init__()

        self.l_emb = nn.Embedding(num_classes, embed_dim)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128 * 4 * 4 + embed_dim, 1)

    def forward(self, x, labels):
        l_emb = self.l_emb(labels)

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn1(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.2)

        x = x.view(x.size(0), -1)
        x = torch.cat([x, l_emb], dim=1)
        x = torch.sigmoid(self.fc(x))

        return x.squeeze()

def loss_function():
    return nn.BCELoss()

