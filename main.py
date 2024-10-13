from utils import parse_gz_file, download_dataset
import pandas as pd
import logging
from train import trainQEModel

# Extract the source and target sentences
dataset_url = "https://web-language-models.s3.amazonaws.com/paracrawl/release9/en-ga/en-ga.txt.gz"
output_file_path = "Dataset/en-ga.txt.gz"
logging.disable(logging.WARNING)

# Download the dataset if not already downloaded
download_dataset(dataset_url, output_file_path)

source_sentences, target_sentences = parse_gz_file(output_file_path)

# Create a DataFrame
paracrawlDataset = pd.DataFrame({
    'source': source_sentences,
    'target': target_sentences
})
paracrawlDataset = paracrawlDataset[:100_000]

paracrawlDataset.to_csv('Dataset/paracrawl_en_ga.csv', index=False)

trainQEModel(paracrawlDataset, batch_size=64)