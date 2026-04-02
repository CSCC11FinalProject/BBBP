# Random Forest (RF) for Blood-Brain Barrier Permeability (BBBP) Prediction
## Installation and Prerequisites
To get started, create a new conda environment and install the core libraries using `pip install -r requirements.txt`. Ensure that `rdkit`, `scikit-learn`, and `pandas` are included in your environment.

## Project Structure
* `rf_model.py`: The core script. It handles data loading, Morgan Fingerprint extraction, Grid Search hyperparameter tuning, model training, and evaluation.

## Model Architecture Summary
* **Feature Engineering**: Processes high-dimensional 2048-bit Morgan Fingerprints and key physicochemical descriptors (LogP and TPSA) directly from the dataset.
* **Ensemble Learning**: Utilizes the bagging method to extract structural patterns without overfitting to the training data.
* **Class Balancing**: Applies a `class_weight='balanced'` parameter to natively counteract the severe 75/25 class imbalance of the dataset during splitting.

## How to Use
### 1. Finding Hyperparameters
Hyperparameter tuning is built into the script using `GridSearchCV`. You can modify the `param_grid` to try new hyperparameters.

### 2. Training the Model
Run the script using `python rf_model.py`. The script will automatically execute the stratified 75/10/15 data split, set up the cross-validation folds, and train the Random Forest on the optimal hyperparameter grid. 

### 3. Evaluating Performance
Automatically evaluate after training. The script will output the optimal hyperparameters from the `param_grid`, AUC-ROC, F1-Score, and a full classification report. Then it will automatically perform false-positive error analysis and save all resulting visualizations and CSVs to your current workspace directory.

## Results
* **Precision**: 0.889
* **F1-Score**: 0.922
* **AUC-ROC**: 0.911
