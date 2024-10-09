import pandas as pd

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

from prdc import compute_prdc



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

class CSVImageDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None, limit=None):
        self.data = pd.read_csv(csv_file)
        print("Before filtered length : ", len(self.data))
        self.data = self.data[self.data['Metadata_CPD_NAME'] != 'dmso']  # Exclude DMSO
        print("After filtered length : ", len(self.data))
        self.image_folder = image_folder
        self.transform = transform
        
        if limit:
            self.data = self.data.head(limit)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_key = self.data.iloc[idx]['SAMPLE_KEY']
        img_path = os.path.join(self.image_folder, f"{sample_key}.png")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Returning a dummy label

# Updated function to use CSV dataset
def calculate_metrics_from_csv(gt_csv, image_folder, fake_folder, batch_size=64, num_workers=4, dims=2048, nearest_k=10, use_cuda=True):
    # Data loading
    transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    
    real_dataset = CSVImageDataset(csv_file=gt_csv, image_folder=image_folder, transform=transform)
    fake_dataset = SimpleImageDataset(fake_folder, transform=transform)
    
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Initialize InceptionV3 model
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]])
    model.eval()
    model.to('cuda' if use_cuda else 'cpu')
    
    real_features, fake_features = [], []
    
    # Extract features for both datasets
    with torch.no_grad():
        print("Loading real images from CSV")
        for real_batch in tqdm(real_loader):
            real_batch = real_batch[0].to('cuda' if use_cuda else 'cpu')
            real_features.append(model(real_batch)[0].view(real_batch.size(0), -1).cpu().numpy())
        
        print("Loading fake images")
        for fake_batch in tqdm(fake_loader):
            fake_batch = fake_batch[0].to('cuda' if use_cuda else 'cpu')
            fake_features.append(model(fake_batch)[0].view(fake_batch.size(0), -1).cpu().numpy())
    
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    
    # FID Calculation
    fid_value = cal_frechet_distance(np.mean(real_features, axis=0), np.cov(real_features, rowvar=False),
                                     np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False))
    
    # Density and Coverage Calculation
    metrics = compute_prdc(real_features, fake_features, nearest_k)
    
    return {'FID': fid_value, **metrics}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate metrics between real and fake images using a CSV for the ground truth.')
    
    parser.add_argument('--gt_csv', type=str, required=True, 
                        help='Path to the CSV file containing ground truth image metadata.')
    parser.add_argument('--image_folder', type=str, required=True, 
                        help='Folder where real images are stored (combined/SAMPLE_KEY.png).')
    parser.add_argument('--gen_folder', type=str, required=True, 
                        help='Path to the folder containing fake images.')

    args = parser.parse_args()

    results = calculate_metrics_from_csv(gt_csv=args.gt_csv, 
                                         image_folder=args.image_folder, 
                                         fake_folder=args.gen_folder)

    print(results)
