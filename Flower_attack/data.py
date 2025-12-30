import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
from config import BATCH_SIZE, SHUFFLE, IMG_SHAPE

class SingleImageDataset(Dataset):
    """Wrap a single image as a Dataset"""
    def __init__(self, image_path, transform=None, label=0):
        self.img = Image.open(image_path).convert("L")  # convert to grayscale
        self.transform = transform
        self.label = label

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = self.img
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.label, dtype=torch.long)


def prepare_dataloader(path="./MNIST", use_selfie=False, selfie_path=None, selfie_label=0):
    transform = transforms.Compose([
        transforms.Resize(IMG_SHAPE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if use_selfie:
        assert selfie_path is not None, "Must provide a selfie_path if use_selfie=True"
        dataset = SingleImageDataset(selfie_path, transform=transform, label=selfie_label)
    else:
        assert BATCH_SIZE == 1, "Gradient inversion requires batch_size=1"
        assert SHUFFLE is False
        dataset = torchvision.datasets.MNIST(
            root=path,
            train=True,
            download=True,
            transform=transform
        )

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=0)
