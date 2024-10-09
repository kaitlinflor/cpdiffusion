from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
import torch.nn.functional as F
from typing import Optional
from transformers import PreTrainedTokenizer
import json
import sys, os
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

clip_module_path = 'utils/cloome/src'
sys.path.append(clip_module_path)
from clip.model_transformers import CLIPGeneral, CLIPGeneralConfig
from clip.clip import _transform

DMSO = "CS(=O)C"

class CustomStableDiffusionPipeline(StableDiffusionPipeline):
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        # Your custom implementation with cloome=True
        cloome = True
        if cloome:
            if lora_scale is not None and isinstance(self, LoraLoaderMixin):
                self._lora_scale = lora_scale

                # dynamically adjust the LoRA scale
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            inputs = self.tokenizer(
                prompt
            )

            encoder_hidden_states = self.text_encoder.encode_text(text=inputs["input_ids"].to(device))
            prompt_embeds = pad_embeddings(encoder_hidden_states, 768).unsqueeze(1).to(device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds_dtype = self.unet.dtype
            prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            # duplicate text embeddings for each generation per prompt
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            # negative_prompt = None
            # if negative_prompt: 
            #     i = 1
            
            # else: 
            #     # HAVE I BEEN DOING NEGATIVE PROMPT WRONG THIS ENTIRE TIME???!!!
            #     negatives = torch.zeros_like(inputs["input_ids"]).to(device)
            #     negative_embeds = pad_embeddings(encoder_hidden_states, 768).unsqueeze(1).to(device)
            #     negative_embeds = negative_embeds.to(dtype=prompt_embeds_dtype, device=device)

            # negative_inputs = torch.zeros_like(inputs["input_ids"]).to(device)
            # negative_inputs = self.tokenizer(
            #     [DMSO], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            # )


            ## INVERTED ##
            # negative_inputs = self.tokenizer(
            #     prompt, invert=True
            # )

            # negative_hidden_states = self.text_encoder.encode_text(text=negative_inputs["input_ids"].to(device))

            ## ALL ZEROS FINGERPRINT 
            negative_inputs_shape = inputs["input_ids"].shape
            negative_inputs_tensor = torch.zeros(negative_inputs_shape, dtype=torch.float32).to(device)

            negative_hidden_states = self.text_encoder.encode_text(text=negative_inputs_tensor)

            negative_embeds = pad_embeddings(negative_hidden_states, 768).unsqueeze(1).to(device)
            # negative_embeds = pad_embeddings(encoder_hidden_states, 768).unsqueeze(1).to(device)

            negative_embeds = negative_embeds.to(dtype=prompt_embeds_dtype, device=device)

            # TRYING JUST 0 
            negative_embeds = torch.ones_like(prompt_embeds).to(device)

            # print(f"Prompt Embeds Device: {prompt_embeds.device}")
            # print(f"Negative Prompt Embeds Device: {negative_prompt_embeds.device}")
            # print("POS EMBEDS : ")
            # print(prompt_embeds)
            # print(f"Min value of prompt_embeds: {torch.min(prompt_embeds).item()}")
            # print(f"Max value of prompt_embeds: {torch.max(prompt_embeds).item()}")

            # print("NEG EMBEDS : ")
            # print(negative_embeds)
            # print(f"Min value of neg: {torch.min(negative_embeds).item()}")
            # print(f"Max value of neg: {torch.max(negative_embeds).item()}")

            # return prompt_embeds, negative_embeds
            # return prompt_embeds, prompt_embeds
            return prompt_embeds, prompt_embeds

        # Default implementation when cloome=False
        return super().encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance,
                                     negative_prompt, prompt_embeds, negative_prompt_embeds, lora_scale, clip_skip)



class CustomStableDiffusionImg2ImgPipeline(StableDiffusionImg2ImgPipeline):
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        # Your custom implementation with cloome=True
        cloome = True
        if cloome:
            if lora_scale is not None and isinstance(self, LoraLoaderMixin):
                self._lora_scale = lora_scale

                # dynamically adjust the LoRA scale
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            inputs = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )

            encoder_hidden_states = self.text_encoder.encode_text(text=inputs["input_ids"].to(device))
            prompt_embeds = pad_embeddings(encoder_hidden_states, 768).unsqueeze(1).to(device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds_dtype = self.unet.dtype
            prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            # duplicate text embeddings for each generation per prompt
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            # negatives = torch.zeros_like(inputs["input_ids"]).to(device)

            # negatives = 
            encoder_hidden_states = self.text_encoder.encode_text(text=inputs["input_ids"].to(device))
            negative_embeds = pad_embeddings(encoder_hidden_states, 768).unsqueeze(1).to(device)
            negative_embeds = negative_embeds.to(dtype=prompt_embeds_dtype, device=device)

            return prompt_embeds, negative_embeds

        # Default implementation when cloome=False
        return super().encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance,
                                     negative_prompt, prompt_embeds, negative_prompt_embeds, lora_scale, clip_skip)


class CLOOMETokenizer(PreTrainedTokenizer):
    def __init__(self, radius=3, nBits=1024, chiral=True):
        super().__init__()
        self.radius = radius
        self.nBits = nBits
        self.chiral = chiral
        self.model_max_length = nBits

    def __call__(self, smiles_list, return_tensors="pt", max_length=None, padding=None, truncation=True, invert=False):
        if invert:  
            fingerprints = [self.inverted_morgan_from_smiles(smiles) for smiles in smiles_list]

        else:
            fingerprints = [self.morgan_from_smiles(smiles) for smiles in smiles_list]
        
        tensor_fingerprints = torch.tensor(fingerprints, dtype=torch.float32)
        return {"input_ids": tensor_fingerprints}

    def tokenize(self, smiles, invert=False):
        return self.__call__([smiles], invert=invert)
    
    def morgan_from_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.nBits, useChirality=self.chiral)
        arr = np.zeros((self.nBits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def inverted_morgan_from_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.nBits, useChirality=self.chiral)
        arr = np.zeros((self.nBits,), dtype=np.int8)
        print("Normal arr : ", arr)
        DataStructs.ConvertToNumpyArray(fp, arr)
        inverted_arr = 1 - arr  # Invert the bits
        print("Inverted arr : ", inverted_arr)
        return inverted_arr

    def batch_decode(self, tokens):
        pass

    def decode(self, token):
        pass

    def get_vocab(self):
        return {}   

def pad_embeddings(embeddings, target_dim):
    pad_size = target_dim - embeddings.size(-1)
    if pad_size > 0:
        embeddings = F.pad(embeddings, (0, pad_size), 'constant', 0)
    return embeddings

# Load the CLOOME checkpoint and model
def load(model_path, device, model, image_resolution):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint["state_dict"]

    src_path = "utils/cloome/src"

    model_config_file = os.path.join(src_path, f"training/model_configs/{model.replace('/', '-')}.json")
    print('Loading model from', model_config_file)
    assert os.path.exists(model_config_file)
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    config = CLIPGeneralConfig(**model_info)
    model = CLIPGeneral(config)

    model.to(device)
    model.eval()

    new_state_dict = {k[len('module.'):]: v for k,v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    return model, _transform(image_resolution, image_resolution, is_train=False)