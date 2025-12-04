# Hand Gestures Classifier

A deep learning project for classifying hand gestures using multiple model architectures.

## Project Structure

```
hand-gestures-classifier/
├── data/                          # Dataset (not tracked in git)
├── data_loaders/                  # PyTorch Dataset implementations
├── model_classes/                 # Model architectures
├── training_scripts/              # Jupyter notebooks for training
├── final_models/                  # Saved model checkpoints (not tracked)
├── papers&reports/                # Related papers and reports
├── test_models.ipynb              # Evaluation notebook for all models
├── create_cropped_dataset.py      # Script to generate cropped images
├── pyproject.toml                 # Project dependencies (uv/pip)
└── .gitignore
```

## Setup

```bash
# Clone the repository
git clone https://github.com/quiet98k/hand-gestures-classifier.git
cd hand-gestures-classifier

# Install dependencies (using uv)
uv sync
```
