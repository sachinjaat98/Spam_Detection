import os
import logging
import json
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import yaml
from torch import nn
from transformers import BertTokenizer, BertModel
# /C:/Users/sachi/OneDrive/Desktop/LLM/Spam_Detection/src/Model_eval.py


# -------------------------
# Logging setup
# -------------------------
# logging configuration
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
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



# -------------------------
# Data helpers
# -------------------------
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




# -------------------------
# Evaluation
# -------------------------
def evaluate(model, tokenizer, texts, labels_int, device, batch_size=32):
    model.to(device)
    model.eval()

    # tokenize all at once (simple)
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels_tensor = torch.tensor(labels_int, dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    loader = DataLoader(dataset, batch_size=batch_size)

    preds = []
    probs = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            b_input_ids, b_attn, b_labels = [x.to(device) for x in batch]
            outputs = model(input_ids=b_input_ids, attention_mask=b_attn)
            logits = outputs.logits
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            batch_preds = np.argmax(batch_probs, axis=-1).tolist()
            preds.extend(batch_preds)
            probs.extend(batch_probs.tolist())
            trues.extend(b_labels.cpu().numpy().tolist())
    return trues, preds, probs

from torch import amp
# Example usage of evaluation function
# Assuming model and tokenizer are already loaded, and test_dataloader is prepared
def example_evaluation(model, dataloader, device):
    model.to(device)
    model.eval()

    y_pred = []
    y_true = []
    probs_all = []

    # Iterate through the test dataloader
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with amp.autocast(device_type=device.type):

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs
                
                #use sigmoid for binary classification
                batch_probs = torch.sigmoid(logits).cpu().numpy()
                #convert probabilities to binary predictions
                preds = (batch_probs > 0.5).astype(int).flatten().tolist()
            

            y_pred.extend(preds)
            y_true.extend(labels.cpu().numpy().tolist())
            probs_all.extend(batch_probs.tolist())

    return y_true, y_pred, probs_all

class CustomBERT(nn.Module):
    def __init__(self, num_labels):
        super(CustomBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.linear(pooled_output)
        return logits

def main():
 
    # loading saved bert model .pth file
    logger.info("Loading tokenizer and model...")
    model_dir = "models/model_bert.pth"

    #tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = CustomBERT(num_labels=1)
    model.load_state_dict(torch.load(model_dir))
    model.to('cuda')
    logger.info("Tokenizer and model loaded.")

    # load test data
    logger.info("Loading test dataset...")
    data_path = "data/processed/test_processed.csv"

    #data/processed/ contains test_processed.csv which contains input_ids and target column
    #there is also test_masks.pt file in the same folder

    test_data = pd.read_csv(data_path)
    test_masks = torch.load("data/processed/test_masks.pt")
    texts = test_data.iloc[:, :-1].values
    labels = test_data['target'].values

    # prepare dataloader
    input_ids = torch.tensor(texts, dtype=torch.long)
    attention_masks = test_masks
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    test_dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #testing the model
    logger.info("Evaluating the model...")
    batch_size = 32
    #example_evaluation(model, test_loader, device)



    # evaluate
    logger.info("Starting evaluation...")

    trues, preds, probs = example_evaluation(model, test_loader, device)

    # compute metrics
    acc = accuracy_score(trues, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(trues, preds, average="weighted", zero_division=0)

    # detailed classification report
    cm = confusion_matrix(trues, preds)
    class_report = classification_report(trues, preds, zero_division=0)

    results = {
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "num_samples": len(trues),
        "confusion_matrix": cm.tolist(),
    }

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision (weighted): {precision:.4f}")
    logger.info(f"Recall (weighted): {recall:.4f}")
    logger.info(f"F1 (weighted): {f1:.4f}")
    logger.info("Classification report:\n" + class_report)

    # save results
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


    logger.info(f"Saving evaluation results to {output_dir}")

    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()