import torch
import argparse
from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from utils.sd_pipes import CustomStableDiffusionImg2ImgPipeline
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import json
import os, sys
from transformers import PreTrainedTokenizer
import pandas as pd
from tqdm import tqdm
from inference_cloome import CLOOMETokenizer, pad_embeddings, load
from PIL import Image
import re
import torch
from torchvision import transforms



def main(model_path, chemical_path, save_dir, num_images, checkpoint, strength, dmso_image_path):
    smiles_df = pd.read_csv(chemical_path)
    if 'SMILES' in smiles_df.columns:
        smiles_list = smiles_df['SMILES'].unique()
    elif 'Metadata_SMILES' in smiles_df.columns:
        smiles_list = smiles_df['Metadata_SMILES'].unique()
    else:
        smiles_list = None

    device = torch.device("cuda")

    FILENAME = "cloome-retrieval-zero-shot.pt"
    REPO_ID = "anasanchezf/cloome"
    checkpoint_path = hf_hub_download(REPO_ID, FILENAME)
    text_encoder, _ = load(checkpoint_path, device, "RN50", 520)
    tokenizer = CLOOMETokenizer()

    if checkpoint: 
        unet_path = f"{model_path}/checkpoint-{checkpoint}/unet"
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)
        pipe = CustomStableDiffusionImg2ImgPipeline.from_pretrained(
            model_path, unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, torch_dtype=torch.float16, safety_checker=None
        )
    else:
        pipe = CustomStableDiffusionImg2ImgPipeline.from_pretrained(
            model_path, text_encoder=text_encoder, tokenizer=tokenizer, torch_dtype=torch.float16, safety_checker=None
        )

    pipe.to(device)

    # Load DMSO image as the initial image for img2img
    # init_image = Image.open(dmso_image_path).convert("RGB")
    init_image = Image.open(dmso_image_path).convert("RGB")
    init_image = init_image.resize((96, 96))  # Resize to match your expected image dimensions
    
    # transform = transforms.ToTensor()
    # init_image = transform(init_image).unsqueeze(0)  # Add batch dimension

    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Converts to [0, 1] range
    #     ])
    # init_image = transform(init_image).unsqueeze(0)

    # print(f"The type of init_image is: {type(init_image)}")
    # exit()

    if not isinstance(init_image, (Image.Image, np.ndarray, torch.Tensor)):
        raise ValueError("Initial image is not in the correct format. It must be a PIL Image, numpy array, or torch tensor.")

    # Create save directories
    separate_dir = os.path.join(save_dir, "separate")
    combined_dir = os.path.join(save_dir, "combined")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(separate_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    metadata_list = []

    # List of channel names
    channel_names = ['Actin', 'Tubulin', 'DAPI']

    for smiles in tqdm(smiles_list):
        if 'Metadata_CPD_NAME' in smiles_df.columns:
            cpd_name = smiles_df.loc[smiles_df['Metadata_SMILES'] == smiles, 'Metadata_CPD_NAME'].values[0]
            cpd_name = re.sub(r'[\/\\]', '_', cpd_name)
        
        for i in range(num_images):
            # Tokenize SMILES for text-to-image conditioning
            tokenized_prompt = tokenizer.tokenize(smiles)
            tokenized_prompt["input_ids"] = tokenized_prompt["input_ids"].to(device)
            encoder_hidden_states = text_encoder.encode_text(text=tokenized_prompt["input_ids"])
            encoder_hidden_states = pad_embeddings(encoder_hidden_states, 768).unsqueeze(1)
            
            # Perform img2img generation with strength to control modification
            image = pipe(prompt=[smiles], image=init_image, strength=strength, height=96, width=96).images[0]

            # Save the combined image
            image.save(f"{combined_dir}/{cpd_name}_{i}.png")

            # Separate and save channel images (assuming RGB mapping to channels)
            combined_image = np.array(image)
            separate_images = {
                "Actin": combined_image[:, :, 0],  # Assuming channel 0 is Actin
                "Tubulin": combined_image[:, :, 1],  # Assuming channel 1 is Tubulin
                "DAPI": combined_image[:, :, 2]  # Assuming channel 2 is DAPI
            }

            for key, img in separate_images.items():
                Image.fromarray(img).save(f"{separate_dir}/{cpd_name}_{key}_{i}.png")

            # Store metadata for tracking
            metadata = {
                "Metadata_SMILES": smiles,
                "Metadata_CPD_NAME": cpd_name,
                "PathName_Actin": separate_dir,
                "FileName_Actin": f"{cpd_name}_Actin_{i}.png",
                "PathName_Tubulin": separate_dir,
                "FileName_Tubulin": f"{cpd_name}_Tubulin_{i}.png",
                "PathName_DAPI": separate_dir,
                "FileName_DAPI": f"{cpd_name}_DAPI_{i}.png"
            }

            metadata_list.append(metadata)

    # Save metadata to a CSV file
    metadata_path = os.path.join(save_dir, "metadata.csv")
    pd.DataFrame(metadata_list).to_csv(metadata_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion img2img")
    parser.add_argument("--model_path", type=str, default="cloome-bbbc", help="Path to the saved model")
    parser.add_argument("--chemicals_path", type=str, default="/gscratch/aims/kflores3/cellpainting/data/preprocessed_data/bbbc021/train_metadata.csv", help="File with testing chemicals")
    parser.add_argument("--save_dir", type=str, default="generated_img2img")
    parser.add_argument("--num_images", type=int, default=2, help="Number of images to generate per chemical")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint number to load the UNet model from")
    parser.add_argument("--strength", type=float, default=0.5, help="The strength of the img2img transformation")
    parser.add_argument("--dmso_image_path", default="/gscratch/aims/kflores3/cellpainting/data/preprocessed_data/bbbc021/train/combined/Week1_22381_1_3203_127.0.png", type=str, help="Path to the DMSO image for img2img initialization")

    args = parser.parse_args()

    main(args.model_path, 
         args.chemicals_path, 
         args.save_dir,
         args.num_images,
         args.checkpoint,
         args.strength,
         args.dmso_image_path)
