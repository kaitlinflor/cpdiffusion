import copy
import os
import pandas as pd
import requests
from io import BytesIO
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np
from tqdm import tqdm

from PIL import Image

# from rdkit import Chem
# from rdkit.Chem import AllChem
import pubchempy as pcp

# from torchvision.transforms import Normalize

# If set to true, we will also set original images (3 or 5 channels) as grayscale .png
SAVE_ORIG_IMAGES = True
# If set to true, the original images will be processed as well, similar to RGB image
NORM_ORIG_IMAGES = True
# This is  a path where to save the preprocessed data. Will be used in the metadata
# .csv fiile, that provides path to images.
OUTPUT_BASE_PATH = "preprocessed_data"

s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

def download_image_from_s3(image_url, download_path):
    # Extract the bucket name and key from the image_url
    bucket_name = image_url.split('/')[2]
    key = '/'.join(image_url.split('/')[3:])
    
    # Download the image
    s3_client.download_file(bucket_name, key, download_path)
    # print(f"Image downloaded to {download_path}")


def download_image(s3_client, path, filename):
    bucket_name = path.split('/')[2]
    key = '/'.join(path.split('/')[3:]) + filename
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    return mpimg.imread(BytesIO(response["Body"].read()), format="tiff")

def normalize(channel):
    return (channel - np.min(channel)) / (np.max(channel) - np.min(channel))

# def get_image_from_well(s3_client, load_data_well):
#     channel_paths = {
#         "AGP": load_data_well['PathName_OrigAGP'].values[0],
#         "DNA": load_data_well['PathName_OrigDNA'].values[0],
#         "ER": load_data_well['PathName_OrigER'].values[0],
#         "Mito": load_data_well['PathName_OrigMito'].values[0],
#         "RNA": load_data_well['PathName_OrigRNA'].values[0]
#     }
#
#     channel_files = {
#         "AGP": load_data_well['FileName_OrigAGP'].values[0],
#         "DNA": load_data_well['FileName_OrigDNA'].values[0],
#         "ER": load_data_well['FileName_OrigER'].values[0],
#         "Mito": load_data_well['FileName_OrigMito'].values[0],
#         "RNA": load_data_well['FileName_OrigRNA'].values[0]
#     }
#
#     # Download and normalize each channel
#     images = {}
#     for channel, path in channel_paths.items():
#         filename = channel_files[channel]
#         images[channel] = download_image(s3_client, path, filename)
#         # Normalize the image to [0, 1]
#         images[channel] = normalize(images[channel])
#
#     # Combine the channels into an RGB image
#     combined_image = np.zeros((images["DNA"].shape[0], images["DNA"].shape[1], 3))
#
#     # Map each channel to RGB (example mapping)
#     combined_image[..., 0] = images["DNA"]  # Red channel
#     combined_image[..., 1] = images["AGP"]  # Green channel
#     combined_image[..., 2] = images["ER"]  # Blue channel
#
#     return combined_image


def crop_stack(combined_image):
    height, width, channels = combined_image.shape

    start_x = (width - 696) // 2
    end_x = start_x + 696
    start_y = (height - 520) // 2
    end_y = start_y + 520

    cropped_image = combined_image[start_y:end_y, start_x:end_x, :]

    return cropped_image


def get_chemical_info(chemid):
    try: 
        compound = pcp.get_compounds(chemid, 'inchikey')[0]
        smiles = compound.canonical_smiles
        iupac_name = compound.iupac_name if 'iupac_name' in compound.to_dict() else 'No description available'

        return {"INCHIKEY": chemid, 
                "SMILES": smiles, 
                "CPD_NAME": iupac_name}
    except Exception as e:
        return None

def illumination_threshold(arr, perc=0.0028):
    """ Return threshold value to not display a percentage of highest pixels"""
    perc/=100
    total_pixels = arr.size
    n_pixels = int(np.around(total_pixels * perc))
    flat_inds = np.argpartition(arr, -n_pixels, axis=None)[-n_pixels:]
    threshold = arr.flat[flat_inds].min()
    return threshold

def sixteen_to_eight_bit(arr, display_max, display_min=0):
    threshold_image = ((arr.astype(float) - display_min) * (arr > display_min))
    scaled_image = np.clip(threshold_image * (255. / (display_max - display_min)), 0, 255)
    return scaled_image.astype(np.uint8)


def process_image(arr):
    arr=arr.astype(np.float32)
    threshold = illumination_threshold(arr)

    processed_image = sixteen_to_eight_bit(arr, threshold)

    return processed_image

def normalize_image(image, min_vals, max_vals):
    normalized_image = (image - min_vals) / (max_vals - min_vals)
    return (normalized_image * 255).astype(np.uint8)

def normalize(channel):
    return (channel - np.min(channel)) / (np.max(channel) - np.min(channel))

def save_tiff(image_bytearray, output_file):
    # # Step 1: Create a BytesIO object from the byte array
    # image_stream = BytesIO(image_bytearray)
    # # Step 2: Open the image using Pillow
    # image = Image.open(image_stream)
    # # Step 3: Save the image to a file
    # image.save(output_file)
    with open(output_file, 'wb') as file:
        file.write(image_bytearray)
def process_site(df, chemical_info, num_channels=5):

    channel_paths = {
        "Ph_golgi": df['PathName_OrigAGP'],
        "Hoecsht": df['PathName_OrigDNA'],
        "ERSyto": df['PathName_OrigER'],
        "Mito": df['PathName_OrigMito'],
        "ERSystoBleed": df['PathName_OrigRNA']
    }

    channel_files = {
        "Ph_golgi": df['FileName_OrigAGP'],
        "Hoecsht": df['FileName_OrigDNA'],
        "ERSyto": df['FileName_OrigER'],
        "Mito": df['FileName_OrigMito'],
        "ERSystoBleed": df['FileName_OrigRNA']
    }

    # Metadata for this particular entry. More entries may be updated later in this function.
    metadata = {
        "PLATE_ID": df["Metadata_Plate"],
        "WELL_POSITION": df["Metadata_Well"],
        "SITE": row["Metadata_Site"],
        "SMILES": chemical_info["SMILES"],
        "INCHIKEY": chemical_info["INCHIKEY"],
        "CPD_NAME": chemical_info["CPD_NAME"],
        # TODO*(alexta): below we duplicate metadata columns in the format suitable for CellProfiler
        # These metrics identify the cell site. CellProfiler automatically recognizes them by "Metadata_" prefix and
        # includes these in the output data,
        # in future, it would be nice to remove duplicate columns in our metadata.
        "Metadata_Plate": df["Metadata_Plate"],
        "Metadata_Well": df["Metadata_Well"],
        "Metadata_Site": row["Metadata_Site"],
        "Metadata_InChIKey": row["Metadata_InChIKey"],
    }

    # Download and normalize each channel
    images = {}
    # Raw, unnormalized images from S3
    # images_raw = {}
    for index, (channel, path) in enumerate(channel_paths.items()):
        # print(channel)
        filename = channel_files[channel]
        images[channel] = download_image(s3_client, path, filename)
        # if SAVE_ORIG_IMAGES:
        #     images_raw[channel] = copy.deepcopy(images[channel])
        images[channel] = normalize(images[channel])

    # Mito – Mito
    # AGP – Ph_golgi
    # ER – ERSyto
    # DNA – Hoecsht
    # RNA – ERSystoBleed

    #TODO: There's seem to be a bug - ERSystoBleed and ERSytoBleed are used throughout the file interchangingly
    channels = {'0': 'Mito', '1': 'ERSyto', '2': 'ERSytoBleed', '3': 'Ph_golgi', '4': 'Hoechst'}
    combined_image = np.stack([images["Mito"], images["ERSyto"], images["ERSystoBleed"], images["Ph_golgi"], images["Hoecsht"]], axis=-1)
    combined_image = crop_stack(combined_image)

    processed_channels = [process_image(combined_image[:, :, i]) for i in range(combined_image.shape[-1])]

    combined_image = np.stack(processed_channels, axis=-1)

    id = f"{df['Metadata_Plate']}-{df['Metadata_Well']}-{df['Metadata_Site']}"
    metadata["SAMPLE_KEY"] = id

    if num_channels == 5:
        save_dir = "preprocessed_data/5_channels"
        os.makedirs(save_dir, exist_ok=True)
        np_id = f"{id}.npz"
        np_path = os.path.join(save_dir, np_id)
        np.savez(np_path, sample=combined_image, channels=channels)
        if SAVE_ORIG_IMAGES:
            for (channel_id, raw_image_bytearray) in images.items():
                raw_file_id = f"{id}-{channel_id}.png"
                raw_image = Image.fromarray(process_image(raw_image_bytearray))
                raw_image.save(os.path.join(save_dir, raw_file_id))
                metadata[f"PathName_Orig{channel_id}"] = f"{OUTPUT_BASE_PATH}/5_channels"
                metadata[f"FileName_Orig{channel_id}"] = f"{id}-{channel_id}.png"

    if num_channels == 3:
        R = combined_image[:, :, 4] 
        G = combined_image[:, :, 3]  
        B = combined_image[:, :, 1] 

        rgb_image = np.stack([R, G, B], axis=-1)

        save_dir = "preprocessed_data/3_channels"
        os.makedirs(save_dir, exist_ok=True)
        
        png_id = f"{id}.png"
        save_path = os.path.join(save_dir, png_id)

        image = Image.fromarray(rgb_image)
        image.save(save_path)
        if SAVE_ORIG_IMAGES:
            # for (channel_id, raw_image_bytearray) in images_raw.items():
            for (channel_id, raw_image_bytearray) in images.items():
                raw_file_id = f"{id}-{channel_id}.png"
                raw_image = Image.fromarray(process_image(raw_image_bytearray))
                raw_image.save(os.path.join(save_dir, raw_file_id))
                metadata[f"PathName_Orig{channel_id}"] = f"{OUTPUT_BASE_PATH}/3_channels"
                metadata[f"FileName_Orig{channel_id}"] = f"{id}-{channel_id}.png"

    #
    # metadata = {
    #     "SAMPLE_KEY": id,
    #     "PLATE_ID": df["Metadata_Plate"],
    #     "WELL_POSITION": df["Metadata_Well"],
    #     "SITE": row["Metadata_Site"],
    #     "SMILES": chemical_info["SMILES"],
    #     "INCHIKEY": chemical_info["INCHIKEY"],
    #     "CPD_NAME": chemical_info["CPD_NAME"],
    #     "FileName_OrigAGP",
    #     "FileName_OrigDNA",
    #     "FileName_OrigER",
    #     "FileName_OrigMito",
    #     "FileName_OrigRNA",
    #     "PathName_OrigAGP",
    #     "PathName_OrigDNA",
    #     "PathName_OrigER",
    #     "PathName_OrigMito",
    #     "PathName_OrigRNA",
    # }

    return metadata


if __name__ == "__main__":
    profile_formatter = (
        "s3://cellpainting-gallery/cpg0016-jump/"
        "{Metadata_Source}/workspace/profiles/"
        "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.parquet"
    )

    loaddata_formatter = (
        "s3://cellpainting-gallery/cpg0016-jump/"
        "{Metadata_Source}/workspace/load_data_csv/"
        "{Metadata_Batch}/{Metadata_Plate}/load_data_with_illum.parquet"
    )

    GIT_CLONE_DIR = "./"

    plates = pd.read_csv(os.path.join(GIT_CLONE_DIR, "metadata/plate.csv.gz"))
    wells = pd.read_csv(os.path.join(GIT_CLONE_DIR, "metadata/well.csv.gz"))
    compounds = pd.read_csv(os.path.join(GIT_CLONE_DIR, "metadata/compound.csv.gz"))

    # Whether to sample a certain number of wells per chemical
    sample_from_group = True
    sample_number = 10

    # Whether to use all chemicals or specific ones 
    all_chemicals = False

    # Total number of sites for each chemical
    site_number = 10

    # Get plates with compounds.
    compound_plates = plates[plates['Metadata_PlateType'] == 'COMPOUND']
    compound_plates_with_wells = compound_plates.merge(wells, on=['Metadata_Source', 'Metadata_Plate'])
    # Result: a single table with source, batchID, plate ID, plate type (always "compound"), well ID,
    # and chemical key / checmical structure in SMILES format
    wells_with_chems = compound_plates_with_wells.merge(compounds, on="Metadata_JCP2022")

    # Get rid of NA 
    selected_chemicals = wells_with_chems['Metadata_InChIKey'].dropna().unique()
    wells_with_chems = wells_with_chems[wells_with_chems['Metadata_InChIKey'].isin(selected_chemicals)]

    # Get chemical counts and dataframe with only > counts
    wells_per_chemical = wells_with_chems['Metadata_InChIKey'].value_counts().reset_index()
        # CRITERIA FOR WHICH CHEMICALS TO USE
    if all_chemicals:
        valid_chemicals = selected_chemicals
    else:
        valid_chemicals = wells_per_chemical[(wells_per_chemical['count'] > 10) & (wells_per_chemical['count'] < 19)]['Metadata_InChIKey']

    print("Number of valid chemicals : ", len(valid_chemicals))

    # alexta: sample chemicals to get only 10 of them (Debug)
    valid_chemicals = valid_chemicals.sample(n=10)



    valid_wells = wells_with_chems[wells_with_chems['Metadata_InChIKey'].isin(valid_chemicals)]

    if sample_from_group:
        # Essentially, get a sample of <batch, plate, well> for chemicals; max 10 entries per chemical
        valid_wells = valid_wells.groupby(['Metadata_InChIKey'], group_keys=False).apply(lambda x: x.sample(sample_number))

    # Get plates we need to load : 
    relevant_plates = valid_wells['Metadata_Plate'].unique()
    print("Number of plates : ", len(relevant_plates))

    # Load data for plates
    load_data = []
    for plate in tqdm(relevant_plates):
        # print(plate)
        rows = compound_plates[compound_plates['Metadata_Plate'] == plate]
        row = compound_plates[compound_plates['Metadata_Plate'] == plate].iloc[0]
        s3_path = loaddata_formatter.format(**row.to_dict())
        # HEre we have plenty of info from the S3: the usual "key" (i.e. source, batch, plate, site)
        # Paths to the
        load_data.append(pd.read_parquet(s3_path, storage_options={"anon": True}))
    load_data = pd.concat(load_data)

    list_chemicals = valid_chemicals.to_list()

    # Metadata files for datasets with 3 and 5 channels, respectively
    metadata3 = []
    metadata5 = []
    no_data = 0
    for chemical in tqdm(valid_chemicals):
        # For a given chemical, get it formula in multiple formats
        chemical_info = get_chemical_info(chemical)
        if not chemical_info:
            no_data += 1
            continue

        print("Chemical : ", chemical_info["CPD_NAME"])

        valid_well = valid_wells[valid_wells['Metadata_InChIKey'] == chemical]

        merged_df = pd.merge(
            valid_well,
            load_data,
            on=['Metadata_Source', 'Metadata_Batch', 'Metadata_Plate', 'Metadata_Well'],
            how='inner'
            ).sample(site_number)

        i = 0

        for iter, row in merged_df.iterrows():
            print(f"Processing row {i}")
            i += 1
            row_metadata3 = process_site(row, chemical_info, num_channels=3)
            row_metadata5 = process_site(row, chemical_info, num_channels=5)
            metadata3.append(row_metadata3)
            metadata5.append(row_metadata5)

    metadata_df3 = pd.DataFrame(metadata3)
    metadata_df3.to_csv("preprocessed_data/metadata3.csv", index=False)
    metadata_df5 = pd.DataFrame(metadata5)
    metadata_df5.to_csv("preprocessed_data/metadata5.csv", index=False)

    print("Number of compounds that weren't found : ", no_data)
