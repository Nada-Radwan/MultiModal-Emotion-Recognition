import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from utils.config import Config

class EmotionImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = []
        self.labels = []
        
        transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor()
        ])
        
        self.transform = transform
        
        for label in os.listdir(image_dir):
            label_dir = os.path.join(image_dir, label)
            for img in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img))
                self.labels.append(int(label))
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        
        label = torch.zeros(Config.NUM_LABELS)
        label[self.labels[idx]] = 1
        
        return image, label
