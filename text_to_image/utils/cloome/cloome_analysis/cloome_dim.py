import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from huggingface_hub import hf_hub_download
import os, json
from transformers import PreTrainedTokenizer
import umap
import sys 

clip_module_path = '/gscratch/aims/kflores3/cellpainting/cloome/src'
sys.path.append(clip_module_path)
from clip.model_transformers import CLIPGeneral, CLIPGeneralConfig
from clip.clip import _transform

# Define the tokenizer and padding function for CLOOME
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import torch.nn.functional as F

def morgan_from_smiles(smiles, radius=3, nbits=1024, chiral=True):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits, useChirality=chiral)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

class CLOOMETokenizer(PreTrainedTokenizer):
    def __init__(self, radius=3, nBits=1024, chiral=True):
        super().__init__()
        self.radius = radius
        self.nBits = nBits
        self.chiral = chiral
        self.model_max_length = nBits

    def __call__(self, smiles_list, return_tensors="pt", max_length=None, padding=None, truncation=True):
        fingerprints = [morgan_from_smiles(smiles, self.radius, self.nBits, self.chiral) for smiles in smiles_list]
        tensor_fingerprints = torch.tensor(fingerprints, dtype=torch.float32)
        return {"input_ids": tensor_fingerprints}

    def tokenize(self, smiles):
        return self.__call__([smiles])
    
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

# Load the CLOOME model
def load_cloome_model(model_path, device):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint["state_dict"]

    src_path = "/gscratch/aims/kflores3/cellpainting/cloome/src"

    model_config_file = os.path.join(src_path, f"training/model_configs/RN50.json")
    print('Loading model from', model_config_file)
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)

    config = CLIPGeneralConfig(**model_info)
    model = CLIPGeneral(config)

    model.to(device)
    model.eval()

    new_state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    return model

# Main workflow to extract embeddings and plot
def main():
    # Load metadata
    metadata_path = "/gscratch/aims/kflores3/cellpainting/IMPA_reproducibility/datasets/bbbc021_all/metadata/bbbc021_df_all.csv"
    metadata = pd.read_csv(metadata_path)  # Replace with your actual file path

    # Extract SMILES and Mechanism of Action (ANNOT)
    smiles_to_mechanism = metadata[['SMILES', 'ANNOT']].drop_duplicates()

    print(smiles_to_mechanism)

    # Load the CLOOME model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = hf_hub_download("anasanchezf/cloome", "cloome-retrieval-zero-shot.pt")
    cloome_model = load_cloome_model(checkpoint_path, device)
    tokenizer = CLOOMETokenizer()

    # Get CLOOME embeddings for all SMILES
    embeddings = []
    for smiles in smiles_to_mechanism['SMILES']:
        tokenized_prompt = tokenizer.tokenize(smiles)
        tokenized_prompt["input_ids"] = tokenized_prompt["input_ids"].to(device)
        
        with torch.no_grad():
            embedding = cloome_model.encode_text(text=tokenized_prompt["input_ids"])
            embedding = pad_embeddings(embedding, 768).cpu().numpy().flatten()  # Convert to numpy array and flatten
            embeddings.append(embedding)

    embeddings = np.array(embeddings)


    # embeddings = np.array(embeddings)

        # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot the embeddings, colored by mechanism of action
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(smiles_to_mechanism['ANNOT'].unique()):
        indices = smiles_to_mechanism['ANNOT'] == label
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=label)

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('CLOOME Embeddings with t-SNE')
    plt.legend()
    plt.savefig('cloome_tsne.png')

    # Apply UMAP for dimensionality reduction
    # reducer = umap.UMAP(n_components=2, random_state=42)
    # reduced_embeddings = reducer.fit_transform(embeddings)

    # # Plot the embeddings, colored by mechanism of action
    # plt.figure(figsize=(10, 8))
    # for i, label in enumerate(smiles_to_mechanism['ANNOT'].unique()):
    #     indices = smiles_to_mechanism['ANNOT'] == label
    #     plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=label)

    # plt.xlabel('UMAP Dimension 1')
    # plt.ylabel('UMAP Dimension 2')
    # plt.title('CLOOME Embeddings with UMAP')
    # plt.legend()
    # plt.savefig("cloome_UMAP")

    # # Apply dimensionality reduction (PCA or t-SNE)
    # pca = PCA(n_components=2)
    # reduced_embeddings = pca.fit_transform(embeddings)

    # # Plot the embeddings, colored by mechanism of action
    # plt.figure(figsize=(10, 8))
    # for i, label in enumerate(smiles_to_mechanism['ANNOT'].unique()):
    #     indices = smiles_to_mechanism['ANNOT'] == label
    #     plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=label)

    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.title('CLOOME Embeddings with Dimensionality Reduction')
    # plt.legend()
    # plt.savefig("cloome.png")

if __name__ == "__main__":
    main()
