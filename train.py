import torch
import torch.nn as nn
import torch.optim as optim

from load_data import get_data
from model import Generator, Discriminator, loss_function
from config import (
    Z_DIM, NUM_CLASSES, EMBED_DIM, BATCH_SIZE, EPOCHS, LR, BETA1, BETA2, SAVE_DIR, DEVICE
)


def train():
    G = Generator(z_dim=Z_DIM, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM).to(DEVICE)
    D = Discriminator(num_classes=NUM_CLASSES, embed_dim=EMBED_DIM).to(DEVICE)

    optim_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    optim_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

    criterion = loss_function()
    train_data = get_data(batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        G.train()
        D.train()

        epoch_loss_G = 0
        epoch_loss_D = 0

        for bi, (real_img, labels) in enumerate(train_data):
            batch_size = real_img.size(0)
            real_img = real_img.to(DEVICE)
            labels = labels.to(DEVICE)

            real_labels = torch.ones(batch_size, device=DEVICE)
            fake_labels = torch.zeros(batch_size, device=DEVICE)

            D.zero_grad()
            d_real = D(real_img, labels)
            loss_d_real = criterion(d_real, real_labels)

            z = torch.randn(batch_size, Z_DIM, device=DEVICE)
            fake_img = G(z, labels)
            d_fake = D(fake_img.detach(), labels)
            loss_d_fake = criterion(d_fake, fake_labels)

            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            optim_D.step()


            G.zero_grad()
            z = torch.randn(batch_size, Z_DIM, device=DEVICE)
            fake_img = G(z, labels)
            d_fake = D(fake_img, labels)

            loss_g = criterion(d_fake, real_labels)
            loss_g.backward()
            optim_G.step()

            epoch_loss_D += loss_d.item()
            epoch_loss_G += loss_g.item()

        avg_loss_D = epoch_loss_D / len(train_data)
        avg_loss_G = epoch_loss_G / len(train_data)

        print(f'Epoch {epoch+1:03d}/{EPOCHS} | Loss D: {avg_loss_D:.4f} | Loss G: {avg_loss_G:.4f}')
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
            torch.save(G.state_dict(), f'{SAVE_DIR}/generator_epoch_{epoch+1}.pth')
            torch.save(D.state_dict(), f'{SAVE_DIR}/discriminator_epoch_{epoch+1}.pth')
            print(f'  💾 Веса сохранены: epoch {epoch+1}')
    
    print("✅ Обучение завершено!")

if __name__ == "__main__":
    train()
