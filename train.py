import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from qemodel import ParaCrawlDataset, QEModel

def trainQEModel(df, batch_size, lr=2e-5):
    # Initialize tokenizer (multilingual BERT)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Check if GPU is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Instantiate the model and move it to the GPU if available
    model = QEModel('bert-base-multilingual-cased')
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Load the dataset into DataLoader
    dataset = ParaCrawlDataset(df['source'], df['target'], tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0
        for batch in data_loader:
            print(count)
            count += 1                  
            # Move the batch data to the GPU if available
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).squeeze()
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluation on the test set (example)
    model.eval()
    predictions, true_scores = [], []
    with torch.no_grad():
        for batch in data_loader:
            # Move the batch data to the GPU if available
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask).squeeze()
            predictions.extend(outputs.cpu().numpy())
            true_scores.extend(scores.cpu().numpy())

    # Compute the Mean Squared Error (example evaluation)
    mse = mean_squared_error(true_scores, predictions)
    print(f"Test Mean Squared Error: {mse:.4f}")