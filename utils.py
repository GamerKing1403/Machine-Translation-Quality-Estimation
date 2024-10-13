import os
import requests
import gzip

# Function to extract parallel sentences from a gzipped text file
def parse_gz_file(gz_file):
    print(f"Parsing {gz_file}...")
    source_sentences = []
    target_sentences = []
    
    with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
        for line in f:
            if "\t" in line:  # Split by tab
                source, target = line.strip().split('\t')
                source_sentences.append(source)
                target_sentences.append(target)
    
    print(f"Parsed {len(source_sentences)} sentence pairs.")
    return source_sentences, target_sentences

def download_file(url, output_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"Download completed: {output_path}")
    
def download_dataset(dataset_url, output_file_path):
    if not os.path.exists(output_file_path):
        download_file(dataset_url, output_file_path)
    else:
        print("Dataset already downloaded!")
