import os
from torch.utils.data import Dataset
from PIL import Image


class CelebDataset(Dataset):
    def __init__(self, root_dir="./datasets/Celeb-DF-v2", mode='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        if mode == 'train':
            real_dir = os.path.join(root_dir, "real")
            synthesis_dir = os.path.join(root_dir, "fake")
        elif mode == 'test':
            real_dir = os.path.join(root_dir, "real_test")
            synthesis_dir = os.path.join(root_dir, "fake_test")
        else:
            raise ValueError("mode must be 'train' or 'test'")

        if not os.path.exists(real_dir) or not os.path.exists(synthesis_dir):
            raise ValueError(f"One or both directories {real_dir}, {synthesis_dir} do not exist.")

        for img_name in os.listdir(real_dir):
            img_path = os.path.join(real_dir, img_name)
            self.data.append(img_path)
            self.labels.append(0)

        for img_name in os.listdir(synthesis_dir):
            img_path = os.path.join(synthesis_dir, img_name)
            self.data.append(img_path)
            self.labels.append(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label