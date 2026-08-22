import os
import torch

Z_DIM = 128
NUM_CLASSES = 10
EMBED_DIM = 50
IMG_SIZE = 32
CHANNELS = 3

BATCH_SIZE = 128
EPOCHS = 50
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
NUM_WORKERS = 0 if torch.cuda.is_available() == False else 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "weights")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
