import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml
#utilizing BERT model for feature extraction
from transformers import BertTokenizer, BertModel
import torch

from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
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
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


#applying BERT tokenizer and model to get features
def bert_encode(texts, tokenizer, max_len=512):

    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


def apply_encode(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Tokenizing data."""
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        train_inputs, train_masks = bert_encode(X_train, tokenizer, max_len=64)
        test_inputs, test_masks = bert_encode(X_test, tokenizer, max_len=64)


        logger.debug('BERT encoding applied with max features %d', max_features)


        #return dataframes
        train_df = pd.DataFrame(train_inputs.numpy())
        train_df['target'] = y_train
        test_df = pd.DataFrame(test_inputs.numpy())
        test_df['target'] = y_test


        return train_df, test_df,train_masks, test_masks
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise


#function save processed data here (train_df, test_df, train_masks, test_masks)
def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame, train_masks: torch.Tensor, test_masks: torch.Tensor, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        process_data_path = os.path.join(data_path, 'processed')
        os.makedirs(process_data_path, exist_ok=True)

        train_df.to_csv(os.path.join(process_data_path, "train_processed.csv"), index=False)
        test_df.to_csv(os.path.join(process_data_path, "test_processed.csv"), index=False)

        torch.save(train_masks, os.path.join(process_data_path, "train_masks.pt"))
        torch.save(test_masks, os.path.join(process_data_path, "test_masks.pt"))

        logger.debug('Processed train and test data saved to %s', process_data_path)

    except Exception as e:
        logger.error('Unexpected error occurred while saving the processed data: %s', e)
        raise



def main():
    try:
        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features']
        # max_features = 50

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        #encoding using BERT
        train_df, test_df, train_masks, test_masks = apply_encode(train_data, test_data, max_features)
    
        #logging
        logger.debug('Creating DataLoader sampler types: random for train, sequential for test')

        # Save the processed data
        save_processed_data(train_df, test_df, train_masks, test_masks, data_path='./data')

        


    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()