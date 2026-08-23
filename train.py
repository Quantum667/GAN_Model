import torch
import torch.nn as nn
import torch.optim as optim

from load_data import get_data
from model import Generator, Discriminator
from config import (
    Z_DIM, NUM_CLASSES, EMBED_DIM, BATCH_SIZE, EPOCHS, LR, BETA1, BETA2, SAVE_DIR, DEVICE, N_CRITIC, LAMBDA_GP
)


def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=DEVICE)

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolate = D(interpolates, labels)

    fake = torch.ones_like(real_samples.size(0), 1, device=DEVICE)
    gradients = torch.autograd.grad(
        outputs=d_interpolate,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

def train():
    G = Generator(z_dim=Z_DIM, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM).to(DEVICE)
    D = Discriminator(num_classes=NUM_CLASSES, embed_dim=EMBED_DIM).to(DEVICE)

    optim_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    optim_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

    train_data = get_data(batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        G.train()
        D.train()

        epoch_loss_G = 0
        epoch_loss_D = 0
        epoch_gp = 0

        for bi, (real_img, labels) in enumerate(train_data):
            batch_size = real_img.size(0)
            real_img = real_img.to(DEVICE)
            labels = labels.to(DEVICE)

            for _ in range(N_CRITIC):
                D.zero_grad()

                pred_real = D(real_img, labels)
                loss_real = -torch.mean(pred_real)

                z = torch.randn(batch_size, Z_DIM, device=DEVICE)
                fake_img = G(z, labels)
                pred_fake = D(fake_img, labels)
                loss_fake = torch.mean(pred_fake)

                gp = compute_gradient_penalty(D, real_img, fake_img, labels)

                loss_D = loss_real + loss_fake + LAMBDA_GP * gp
                loss_D.backward()
                optim_D.step()


            G.zero_grad()
            z = torch.randn(batch_size, Z_DIM, device=DEVICE)
            fake_img = G(z, labels)
            pred_fake = D(fake_img, labels)

            loss_G = -torch.mean(pred_fake)
            loss_G.backward()
            optim_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            epoch_gp += gp.item()

        avg_loss_D = epoch_loss_D / len(train_data)
        avg_loss_G = epoch_loss_G / len(train_data)
        avg_gp = epoch_gp / len(train_data)

        print(f'Epoch {epoch+1:03d}/{EPOCHS} | Loss D: {avg_loss_D:.4f} | Loss G: {avg_loss_G:.4f}')
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
            torch.save(G.state_dict(), f'{SAVE_DIR}/generator_epoch_{epoch+1}.pth')
            torch.save(D.state_dict(), f'{SAVE_DIR}/discriminator_epoch_{epoch+1}.pth')
            print(f'  💾 Веса сохранены: epoch {epoch+1}')
    
    print("✅ Обучение завершено!")

if __name__ == "__main__":
    train()
