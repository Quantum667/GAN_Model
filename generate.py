import torch
import matplotlib.pyplot as plt
import os
from model import Generator
from config import Z_DIM, NUM_CLASSES, EMBED_DIM, DEVICE, SAVE_DIR, RESULT_DIR, CLASS_NAMES


def load_generator(epoch=50):
    G = Generator(z_dim=Z_DIM, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM).to(DEVICE)
    
    weights_path = f'{SAVE_DIR}/generator_epoch_{epoch}.pth'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Веса не найдены: {weights_path}")
    
    G.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    G.eval()
    print(f"✅ Генератор загружен: {weights_path}")
    return G


def generate_by_class(class_id, num_images=16, epoch=50, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    G = load_generator(epoch)
    
    z = torch.randn(num_images, Z_DIM, device=DEVICE)
    labels = torch.full((num_images,), class_id, dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        fake_images = G(z, labels)
    
    fake_images = (fake_images + 1) / 2
    
    cols = 4
    rows = num_images // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = fake_images[i].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
    
    class_name = CLASS_NAMES[class_id]
    plt.suptitle(f'Generated: {class_name} (class {class_id})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = f'{RESULT_DIR}/gan_class_{class_id}_{class_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f'✅ Сгенерировано {num_images} изображений класса: {class_name}')
    print(f'💾 Сохранено: {save_path}')


def generate_random(num_images=16, epoch=50, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    G = load_generator(epoch)
    
    z = torch.randn(num_images, Z_DIM, device=DEVICE)
    labels = torch.randint(0, NUM_CLASSES, (num_images,), device=DEVICE)
    
    with torch.no_grad():
        fake_images = G(z, labels)
    
    fake_images = (fake_images + 1) / 2
    
    cols = 4
    rows = num_images // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = fake_images[i].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'{CLASS_NAMES[labels[i].item()]}', fontsize=8)
    
    plt.suptitle('Random Generated Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = f'{RESULT_DIR}/gan_random.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f'✅ Сгенерировано {num_images} случайных изображений')
    print(f'💾 Сохранено: {save_path}')


def interpolate_classes(class_id_1, class_id_2, num_steps=10, epoch=50, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    G = load_generator(epoch)
    
    z = torch.randn(1, Z_DIM, device=DEVICE)
    
    fig, axes = plt.subplots(1, num_steps, figsize=(20, 2))
    
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        
        label_1 = torch.tensor([class_id_1], device=DEVICE)
        label_2 = torch.tensor([class_id_2], device=DEVICE)
        
        emb_1 = G.l_emb(label_1)
        emb_2 = G.l_emb(label_2)
        
        emb_interp = (1 - alpha) * emb_1 + alpha * emb_2
        
        x = torch.cat([z, emb_interp], dim=1)
        x = torch.relu(G.fc(x))
        x = x.view(-1, 256, 4, 4)
        
        x = torch.relu(G.bn1(G.dconv1(x)))
        x = torch.relu(G.bn2(G.dconv2(x)))
        x = torch.relu(G.bn3(G.dconv3(x)))
        x = torch.tanh(G.dconv4(x))
        
        x = (x + 1) / 2
        
        img = x[0].cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'α={alpha:.1f}', fontsize=10)
    
    class_name_1 = CLASS_NAMES[class_id_1]
    class_name_2 = CLASS_NAMES[class_id_2]
    plt.suptitle(f'Interpolation: {class_name_1} → {class_name_2}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = f'{RESULT_DIR}/gan_interp_{class_id_1}_to_{class_id_2}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f'✅ Интерполяция: {class_name_1} → {class_name_2}')
    print(f'💾 Сохранено: {save_path}')


if __name__ == '__main__':
    # Примеры использования (раскомментируй нужное):
    
    # 1. Генерация 16 грузовиков (class 9)
    # generate_by_class(class_id=9, num_images=16, epoch=50, seed=42)
    
    # 2. Случайная генерация
    # generate_random(num_images=16, epoch=50, seed=42)
    
    # 3. Интерполяция между лягушкой (6) и лошадью (7)
    interpolate_classes(class_id_1=6, class_id_2=7, num_steps=10, epoch=50, seed=42)