import torch
import argparse
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
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
from utils.sd_pipes import CustomStableDiffusionPipeline


def main(model_path, chemical_path, save_dir, num_images, checkpoint):
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
        # model_path = "/gscratch/aims/kflores3/cellpainting/diffusers/examples/text_to_image/cell-painting-chems"
        # model_path = "/gscratch/aims/kflores3/cellpainting/text_to_image/cloome-training"
        pipe = CustomStableDiffusionPipeline.from_pretrained(model_path, unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, torch_dtype=torch.float16, safety_checker=None)
        pipe.to(device)

    else:
        pipe = CustomStableDiffusionPipeline.from_pretrained(model_path, text_encoder=text_encoder, tokenizer=tokenizer, torch_dtype=torch.float16, safety_checker=None)
        pipe.to(device)

    print("Save dir : ", save_dir)


    separate_dir = os.path.join(save_dir, "separate")
    combined_dir = os.path.join(save_dir, "combined")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(separate_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    
    metadata_list = []

    channel_names = ['Actin', 'Tubulin', 'DAPI']

    start_processing = False
    print("NUMBER OF CHEMICALS : ", len(smiles_list))
    for smiles in tqdm(smiles_list):
        if 'Metadata_CPD_NAME' in smiles_df.columns:
            cpd_name = smiles_df.loc[smiles_df['Metadata_SMILES'] == smiles, 'Metadata_CPD_NAME'].values[0]

            cpd_name = re.sub(r'[\/\\]', '_', cpd_name)

                    # Check if this is the compound to start processing from
            # if cpd_name == "mevinolin_lovastatin":
                # start_processing = True  # Start processing from here

            # Skip until we reach the desired compound
            # if not start_processing:
                # continue

        # elif 'Metadata_InChIKey' in smiles_df.columns:
        #     inchikey = smiles_df.loc[smiles_df['SMILES'] == smiles, 'Metadata_InChIKey'].values[0]
        # else:
        #     inchikey = None
        # inchikey = smiles_df.loc[smiles_df['SMILES'] == smiles, 'INCHIKEY'].values[0]

        for i in range(num_images):
            tokenized_prompt = tokenizer.tokenize(smiles)
            tokenized_prompt["input_ids"] = tokenized_prompt["input_ids"].to(device)
            encoder_hidden_states = text_encoder.encode_text(text=tokenized_prompt["input_ids"])
            encoder_hidden_states = pad_embeddings(encoder_hidden_states, 768).unsqueeze(1)
            
            # Generate image
            image = pipe(prompt=[smiles], height=96, width=96).images[0]
            image_array = np.array(image)

            # black_pixels = np.all(image_array == [0, 0, 0], axis=-1)
            # black_percentage = np.mean(black_pixels)

            # If the image is more than 95% black, skip to the next iteration
            # if black_percentage > 0.95:
                # continue

            image.save(f"{combined_dir}/{cpd_name}_{i}.png")

            combined_image = np.array(image)

            separate_images = {
                "Actin": combined_image[:, :, 0],  # Assuming channel 0 is Hoecsht
                "Tubulin": combined_image[:, :, 1],  # Assuming channel 1 is Ph_golgi
                "DAPI": combined_image[:, :, 2]  # Assuming channel 2 is ERSyto
            }
            
            for key, img in separate_images.items():
                Image.fromarray(img).save(f"{separate_dir}/{cpd_name}_{key}_{i}.png")
                # Image.fromarray((img * 255).astype(np.uint8)).save(f"{separate_dir}/{inchikey}_{key}_{i}.png")

            metadata = {
                "Metadata_SMILES": smiles,
                "Metadata_CPD_NAME": cpd_name,
                "PathName_Actin": separate_dir,
                "FileName_Actin": f"{cpd_name}_Actin_{i}.png",
                "PathName_Tubulin": separate_dir,
                "FileName_Tubulin": f"{cpd_name}_Tubulin_{i}.png",
                "PathName_DAPI": separate_dir,
                "FileName_DAPI": f"{cpd_name}_DAPI_{i}.png",
                "SPLIT": "train"
                # Add other paths and filenames as needed
            }
        
            metadata_list.append(metadata)

    metadata_path = os.path.join(save_dir, "metadata3.csv")
    pd.DataFrame(metadata_list).to_csv(metadata_path, index=False)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image using Stable Diffusion")
    parser.add_argument("--model_path", type=str, default="/gscratch/aims/kflores3/cellpainting/diffusers/examples/text_to_image/cloome-training", help="Path to the saved model")
    parser.add_argument("--chemicals_path", type=str, default="/gscratch/aims/kflores3/cellpainting/datasets/preprocessed_data/test_metadata.csv", help="File with testing chemicals")
    parser.add_argument("--save_dir", type=str, default="generated_images")
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--checkpoint", type=int, help="Checkpoint number to load the UNet model from")

    args = parser.parse_args()

    main(args.model_path, 
         args.chemicals_path, 
         args.save_dir,
         args.num_images,
         args.checkpoint)
