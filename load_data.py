from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS

def get_data(batch_size = BATCH_SIZE):
    transform = transforms.Compose(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    )

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True
    )
