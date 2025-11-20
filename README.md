# Spam_Detection
Its a BERT based NLP project for spam detection in Emails
## Project overview

A transformer-based email spam detection pipeline using pretrained models (BERT family). The repository includes data preprocessing, model training, evaluation, and inference scripts for classifying emails as spam or not-spam.

## Features

- Tokenization and data pipelines for email text
- Fine-tuning of a pretrained transformer model (BERT)
- Training, validation, and test evaluation metrics (accuracy, precision, recall, F1)
- Model export for inference
- Example inference script for single-email predictions

## Requirements

- Python 3.8+
- PyTorch or TensorFlow (depending on chosen backend)
- Transformers (Hugging Face)
- scikit-learn, pandas, numpy
- Optional: CUDA for GPU acceleration

Install core dependencies:
```
pip install -r requirements.txt
```

## Dataset

- Expect CSV/TSV files with columns: "text" and "label" (label values: 0 = ham, 1 = spam).
- Provide train/val/test splits or use the provided split script to create them.

## Quick start

1. Prepare environment and install requirements.
2. Place dataset files in data/ (see project structure).
3. Run preprocessing:
```
python scripts/preprocess.py --input data/raw.csv --output data/processed.pkl --tokenizer bert-base-uncased
```
4. Train model:
```
python train.py --config configs/train.yaml
```
5. Evaluate:
```
python evaluate.py --checkpoint checkpoints/best_model.pt --data data/test.pkl
```
6. Run inference:
```
python infer.py --model checkpoints/best_model.pt --text "You won a prize! Click here..."
```

## Training & Hyperparameters

- Default training parameters are defined in configs/train.yaml (batch size, epochs, learning rate, scheduler).
- Use gradient accumulation and mixed precision for large batch training on limited GPU memory.
- Save best checkpoints by validation F1.

## Evaluation

- Scripts compute accuracy, precision, recall, F1, and confusion matrix.
- Use threshold tuning if treating outputs as probabilities.

## Inference & Serving

- infer.py provides a simple CLI to classify a single email.
- Export model to TorchScript or ONNX for production serving:
```
python export_model.py --checkpoint checkpoints/best_model.pt --output model.onnx
```

## Project structure

- data/                - raw and processed datasets
- configs/             - training and model configs (YAML)
- checkpoints/         - saved model checkpoints
- scripts/             - preprocessing and utility scripts
- train.py             - training entrypoint
- evaluate.py          - evaluation scripts
- infer.py             - inference CLI
- export_model.py      - model export utilities
- requirements.txt

## Tips

- Inspect class balance and apply weighted loss or up/down sampling if highly imbalanced.
- Use k-fold eval for robust performance estimates.
- Log experiments with TensorBoard or Weights & Biases.

## Contributing

- Fork, create a feature branch, add tests, and open a pull request with a clear description.

## License

Specify an appropriate license in LICENSE (e.g., MIT).

## Contact

For issues or questions, open an issue in this repository.