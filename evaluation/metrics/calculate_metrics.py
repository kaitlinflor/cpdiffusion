from utils.density_and_coverage import compute_d_c
from utils.fid import cal_fid, inception_activations, cal_frechet_distance
from utils.inception import InceptionV3

import torch
from torch.utils.data import Dataset

# from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import numpy as np
import os

from PIL import Image
from tqdm import tqdm
import argparse


class SimpleImageDataset(Dataset):
    def __init__(self, folder, transform=None, limit=None):
        self.folder = folder
        self.transform = transform
        self.image_paths = [os.path.join(folder, fname) for fname in os.listdir(folder) 
                            if os.path.isfile(os.path.join(folder, fname)) and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

        if limit:
            self.image_paths = self.image_paths[:limit]
            

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        # print(f"Loaded image: {img_path}")
        if self.transform:
            image = self.transform(image)
        
        return image, 0 # Returning a dummy label since classes aren't needed# Define the transformation

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

def calculate_metrics(real_folder, fake_folder, batch_size=64, num_workers=4, dims=2048, nearest_k=5, use_cuda=True, custom_channels=None):
    # Data loading
    transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    
    real_dataset = SimpleImageDataset(real_folder, transform=transform)
    fake_dataset = SimpleImageDataset(fake_folder, transform=transform)
    
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Initialize InceptionV3 model
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]])
    model.eval()
    model.to('cuda'if use_cuda else'cpu')
    
    real_features, fake_features = [], []
    
    # Extract features once for both datasets
    with torch.no_grad():
        print("Loading real images")
        for real_batch in tqdm(real_loader):
            real_batch = real_batch[0].to('cuda'if use_cuda else'cpu')
            features = model(real_batch)[0].view(real_batch.size(0), -1).cpu().numpy()
            real_features.append(model(real_batch)[0].view(real_batch.size(0), -1).cpu().numpy())
        
        print("Loading fake images")
        for fake_batch in tqdm(fake_loader):
            fake_batch = fake_batch[0].to('cuda'if use_cuda else'cpu')
            fake_features.append(model(fake_batch)[0].view(fake_batch.size(0), -1).cpu().numpy())
    
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    
    # FID Calculation
    fid_value = cal_frechet_distance(np.mean(real_features, axis=0), np.cov(real_features, rowvar=False),
                                     np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False))
    
    # Density and Coverage Calculation
    metrics = compute_d_c(real_features, fake_features, nearest_k)
    
    return {'FID': fid_value, **metrics}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate metrics between real and fake image folders.')
    
    parser.add_argument('--gt_folder', type=str, required=True, 
                        help='Path to the folder containing real images.')
    parser.add_argument('--gen_folder', type=str, required=True, 
                        help='Path to the folder containing fake images.')

    args = parser.parse_args()

    results = calculate_metrics(real_folder=args.gt_folder, 
                                fake_folder=args.gen_folder)

    print(results)

    # real_folder = '/gscratch/aims/kflores3/cellpainting/preprocessed_data/jump/train/combined'
    # fake_folder = '/gscratch/aims/kflores3/cellpainting/evaluation/cp_analysis/generated_images/combined'