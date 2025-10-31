import os
import numpy as np
import pandas as pd
import pickle
import logging
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import yaml
from transformers import BertTokenizer, BertModel



# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise

    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


class SpamClassifier(nn.Module):
    def __init__(self, bert_model):
        super(SpamClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(bert_model.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_output = self.sigmoid(linear_output)
        return final_output



def train_model(X_train: np.ndarray, y_train: np.ndarray, attention_masks: np.ndarray, params: dict):
    try:
        # Set device
        device = torch.device(params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        logger.debug('Using device: %s', device)

        
        # Model parameters
        batch_size = params.get('batch_size', 32)
        learning_rate = 2e-5
        
        # Convert data to PyTorch tensors
        input_ids = torch.tensor(X_train, dtype=torch.long)
        attention_masks = attention_masks.clone().detach()
        labels = torch.tensor(y_train, dtype=torch.long)
        
        # Create dataset and dataloader
        dataset = TensorDataset(input_ids, attention_masks, labels)
        train_dataloader = DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=batch_size
        )
        
        # Initialize BERT model for sequence classification
        model = BertModel.from_pretrained('bert-base-uncased',num_labels=2)
        model.to(device)
        
        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


        
        model = SpamClassifier(model)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-5)
        loss_fn = nn.BCELoss()
        epochs = 8

        #training loop
        print('Starting training...')
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_attention_mask = batch[1].to(device)
                b_labels = batch[2].to(device).float().unsqueeze(1)
                
                model.zero_grad()
                outputs = model(b_input_ids, b_attention_mask)
                loss = loss_fn(outputs, b_labels)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            avg_train_loss = total_loss / len(train_dataloader)

            print(f'Epoch {epoch + 1}, Loss: {avg_train_loss}')
            logger.debug('Epoch [%d/%d], Average Loss: %.4f', epoch + 1, epochs, avg_train_loss)

        
        return model
        
    except Exception as e:
        logger.error('Error occurred during model training: %s', e)
        raise


       
    

def main():
    try:
        params = load_params('params.yaml')['model_building']
        
        # Load the training data and masks
        train_data = load_data('./data/processed/train_processed.csv')
        train_masks = torch.load('./data/processed/train_masks.pt')
        
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        
        # Train the model
        model = train_model(X_train, y_train, train_masks, params=params)
        
        # Save the trained model
        model_path = 'models/model_bert.pth'
        torch.save(model.state_dict(), model_path)
        logger.debug('Model saved to %s', model_path)
        
        
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        raise

if __name__ == '__main__':
    main()