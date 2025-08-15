import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TerrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset folder.
            transform (callable, optional): Optional transform to apply on images.
        """
        self.root_dir = root_dir
        self.transform = transform

        # List only images ending with "_h.png"
        self.img_files = [
            f for f in os.listdir(root_dir)
            if f.endswith("_h.png")
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_files[idx])
        image = Image.open(img_path).convert("L")  # convert to grayscale if heightmap

        # Crop 28x28 from top-left corner
        image = image.crop((0, 0, 28, 28))

        if self.transform:
            image = self.transform(image)

        return image


# Example usage
from torchvision import transforms
import torch
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),  # converts HxW to CxHxW and scales to [0,1]
    transforms.Normalize((0.5,), (0.5,))  # normalize to [-1, 1] if needed
])

dataset = TerrainDataset(
    root_dir=r"C:\Users\mshestopalov\PycharmProjects\procedural-terrain-generation\data\archive\_dataset",
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Test iteration
for batch in dataloader:
    print(batch.shape)  # should be [batch_size, 1, 28, 28]
    break
