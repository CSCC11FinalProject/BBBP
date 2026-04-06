# MPNN for Blood-Brain Barrier Permeability (BBBP) Prediction
## Installation and Prerequisites
To get started, create a new conda environment and install the core libraries using `pip install -r requirements.txt`/

## Project Structure
* `mpnn.py`: The core `MPNN` class implementation using convolutioal message passing and self-attention pooling.
* `dataloader.py`: Loads data into the model. Handles SMILES-to-graph conversion and extracts 7 physicochemical descriptors.
* `train.py`: Main training script with stratified splitting, class-weighted loss, and early stopping.
* `tuning.py`: Hyperparameter tuning script using Optuna to find the best model configuration
* `evaluate.py`: Evaluation suite for generating test metrics, confusion matrices, and ROC curves. Saves to the `plots/` directory.
* `utils.py`: Utility scripts for processing the dataset and featurizing

## Model Architecture Summary
* **Graph Data**: Employs 4 message-passing steps where node states are updated using a GRU (Gated Recurrence Unit) cell based on neighborhood connectivity.
* **Global Attention**: A Multi-head Attention layer aggregates node-level information into a global graph embedding.
* **Feature Fusion**: The graph embedding is concatenated with 7 tabular physicochemical descriptors before final classification.

## How to Use
### 1. Finding Hyperparametrs
Run the hyperparameter tuning script with `python tuning.py` to identify the optimal configuration for the model.
*Note that the current configuration in `train.py` is already the optimal one, so you may skip this step.*

### 2. Training the Model
Run the training script with `python train.py` to split the dataset, optimize the model, and save the best weights to `checkpoints/best.pt`.

### 3. Evaluating Performance
After training, assess the model on the held-out test set using `python evaluate.py`.
This script outputs the evaluation metrics to the console and generates visualizations (`confusion_matrix.png`, `roc_curve.png`) in the `plots/` directory.

## Results (may vary)
* **Precision**: 0.938
* **F1-Score**: 0.928
* **AUC-ROC**: 0.910