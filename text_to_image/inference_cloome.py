import torch
import argparse
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from utils.sd_pipes import CustomStableDiffusionPipeline, CLOOMETokenizer, pad_embeddings, load
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import json
import os, sys
from transformers import PreTrainedTokenizer

'''
THIS IS THE CLOOME INFERENCE THAT ACTUALLY WORKS
'''


def main(model_path, checkpoint, prompt):
    device = torch.device("cuda")
    
    # Load the custom text encoder and tokenizer
    FILENAME = "cloome-retrieval-zero-shot.pt"
    REPO_ID = "anasanchezf/cloome"
    checkpoint_path = hf_hub_download(REPO_ID, FILENAME)
    text_encoder, _ = load(checkpoint_path, device, "RN50", 520)
    tokenizer = CLOOMETokenizer()
    
    # Tokenize and encode the prompt
    tokenized_prompt = tokenizer.tokenize(prompt)
    tokenized_prompt["input_ids"] = tokenized_prompt["input_ids"].to(device)
    encoder_hidden_states = text_encoder.encode_text(text=tokenized_prompt["input_ids"])
    encoder_hidden_states = pad_embeddings(encoder_hidden_states, 768).unsqueeze(1)
    
    if checkpoint:
        unet_path = f"{model_path}/checkpoint-{checkpoint}/unet"
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)
        # model_path = "/gscratch/aims/kflores3/cellpainting/diffusers/examples/text_to_image/cell-painting-chems"
        pipe = CustomStableDiffusionPipeline.from_pretrained(model_path, unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, torch_dtype=torch.float16, safety_checker=None)
        # pipe = StableDiffusionPipeline.from_pretrained(model_path, unet=unet, text_encoder=text_encoder, torch_dtype=torch.float16)
        pipe.to(device)
    else:
        # Trying this out
        pipe = CustomStableDiffusionPipeline.from_pretrained(model_path, text_encoder=text_encoder, tokenizer=tokenizer, torch_dtype=torch.float16, safety_checker=None, guidance_scale=1.0)
        pipe.to(device)

    # Generate image
    image = pipe(prompt=[prompt], height=96, width=96).images[0]
    image.save("generated_image_0.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image using Stable Diffusion")
    parser.add_argument("--model_path", type=str, default="cloome-bbbc", help="Path to the saved model")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint number to load the UNet model from")
    parser.add_argument("--prompt", type=str, default="C1CN(CCN1)C2=C(C=C3C(=C2)N(C=C(C3=O)C(=O)O)C4=CC=C(C=C4)F)F", help="Prompt for image generation")

    args = parser.parse_args()

    prompt = "CC[C@]1(O)C[C@H]2CN(CCc3c([nH]c4ccccc34)[C@@](C2)(C(=O)OC)c5cc6c(cc5OC)N(C=O)[C@H]7[C@](O)([C@H](OC(=O)C)[C@]8(CC)C=CCN9CC[C@]67[C@H]89)C(=O)OC)C1"
    main(args.model_path, args.checkpoint, prompt)
