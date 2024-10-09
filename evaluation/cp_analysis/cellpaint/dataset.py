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

def download_image_from_s3(image_url, download_path):
    # Extract the bucket name and key from the image_url
    bucket_name = image_url.split('/')[2]
    key = '/'.join(image_url.split('/')[3:])
    
    # Download the image
    s3_client.download_file(bucket_name, key, download_path)
    print(f"Image downloaded to {download_path}")


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

# Define the directory where the metadata files are located
GIT_CLONE_DIR = "./"

# Load metadata
plates = pd.read_csv(os.path.join(GIT_CLONE_DIR, "metadata/plate.csv.gz"))
wells = pd.read_csv(os.path.join(GIT_CLONE_DIR, "metadata/well.csv.gz"))
compounds = pd.read_csv(os.path.join(GIT_CLONE_DIR, "metadata/compound.csv.gz"))

# Filter the plates DataFrame to keep only rows with Metadata_PlateType as COMPOUND
compound_plates = plates[plates['Metadata_PlateType'] == 'COMPOUND']

# print(compound_plates)

# print(wells)
# exit()
compound_plates_with_wells = compound_plates.merge(wells, on=['Metadata_Source', 'Metadata_Plate'])

metadata = compound_plates_with_wells.merge(compounds, on="Metadata_JCP2022")

selected_chemicals = metadata['Metadata_InChIKey'].dropna().unique()
# selected_chemicals = np.random.choice(selected_chemicals, 2000, replace=False)
print("Selected chemicals length : ", len(selected_chemicals))

metadata = metadata[metadata['Metadata_InChIKey'].isin(selected_chemicals)]
# print(metadata)

wells_per_chemical = metadata['Metadata_InChIKey'].value_counts().reset_index()
print("Wells per chemical : ")
valid_chemicals = wells_per_chemical[wells_per_chemical['count'] > 100]['Metadata_InChIKey']
print(valid_chemicals)

metadata = metadata[metadata['Metadata_InChIKey'].isin(valid_chemicals)]

print(metadata)

# exit()
grouped_metadata = metadata.groupby(['Metadata_InChIKey'], group_keys=False).apply(lambda x: x.sample(1))
# grouped_metadata = metadata.groupby(['Metadata_InChIKey'], group_keys=False).apply(lambda x: x.sample(10))

# grouped = metadata.groupby(['Metadata_InChIKey', 'Metadata_Source'], group_keys=False).apply(lambda x: x.sample(1))

relevant_plates = grouped_metadata['Metadata_Plate'].unique()

print("Loading data")
load_data = []
for plate in tqdm(relevant_plates):
    row = compound_plates[compound_plates['Metadata_Plate'] == plate].iloc[0]
    s3_path = loaddata_formatter.format(**row.to_dict())
    load_data.append(pd.read_parquet(s3_path, storage_options={"anon": True}))
load_data = pd.concat(load_data)
print("Done loading data")
print(load_data)


print(grouped_metadata.columns.tolist())

wells_per_chemical_grouped = grouped_metadata['Metadata_InChIKey'].value_counts().reset_index()

selected_inchikey = 'IHLVSLOZUHKNMQ-UHFFFAOYSA-N'


filtered_metadata = grouped_metadata[grouped_metadata['Metadata_InChIKey'] == selected_inchikey]



random_well = filtered_metadata.sample(1).iloc[0]

load_data_well = load_data[(load_data['Metadata_Source'] == random_well['Metadata_Source']) &
                                    (load_data['Metadata_Well'] == random_well['Metadata_Well'])]


image_url = os.path.join(load_data_well['PathName_OrigDNA'].values[0], load_data_well['FileName_OrigDNA'].values[0])

print("Image url : ", image_url)

s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))



download_image_from_s3(image_url, "./downloaded_image.png")


# sample_linked = pd.merge(
#     load_data, sample_profile, on=["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
# )