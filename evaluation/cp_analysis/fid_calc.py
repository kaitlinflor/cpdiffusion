import os
import numpy as np
import torch
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def preprocess_image(image, image_size=299):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

model = inception_v3(pretrained=True, transform_input=False)
model.fc = torch.nn.Identity()  
model.eval()

def get_features(images, model):
    features = []
    with torch.no_grad():
        for img in tqdm(images, desc="Processing images for feature extraction"):  # Add tqdm here
        # for img in images:
            img = preprocess_image(img)
            feature = model(img).numpy()
            features.append(feature)
    features = np.concatenate(features, axis=0)
    return features

def calculate_fid(real_features, generated_features):
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_score = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid_score

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        except UnidentifiedImageError:
            print(f"Skipping file {img_path} as it cannot be identified as an image.")
        # img = Image.open(img_path).convert('RGB')
        # images.append(img)
    return images

test_folder = '/gscratch/aims/kflores3/cellpainting/analysis/preprocessed_data/test/combined'
train_folder = '/gscratch/aims/kflores3/cellpainting/analysis/preprocessed_data/train/combined'
generated_images_folder = '/gscratch/aims/kflores3/cellpainting/diffusers/examples/text_to_image/generated_train/combined'


test_folder = load_images_from_folder(test_folder)
train_folder = load_images_from_folder(train_folder)
real_images = test_folder + train_folder
generated_images = load_images_from_folder(generated_images_folder)

real_features = get_features(real_images, model)
generated_features = get_features(generated_images, model)

fid_score = calculate_fid(real_features, generated_features)
print('FID score:', fid_score)

# FID score : 128.15...? so high bruh