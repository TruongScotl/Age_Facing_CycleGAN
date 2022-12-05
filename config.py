import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TRAIN_DIR = "/Users/macos/Desktop/GAN_Project/CycleGAN/final_data"
# VAL_DIR = "/Users/macos/Desktop/GAN_Project/CycleGAN/final_data"
TRAIN_DIR = "/kaggle/input/cacd2000-train/final_data/"
VAL_DIR = "/kaggle/input/cacd2000-train/final_data"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 2
NUM_EPOCHS = 10
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_O = "/kaggle/working/genh.pth.tar"
CHECKPOINT_GEN_Y = "/kaggle/working/genz.pth.tar"
CHECKPOINT_CRITIC_O = "/kaggle/working/critich.pth.tar"
CHECKPOINT_CRITIC_Y = "/kaggle/working/criticz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)