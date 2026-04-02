# Random Forest (RF) for Blood-Brain Barrier Permeability (BBBP) Prediction
## Installation and Prerequisites
Libraries are stated at the beginning of the corresponding .py code
Only thing you needed is the BBBP.csv file.
Just change the path of the dataset file at the beginning will autiomatically complete all the parts and obtain all the outputs, including the plots.

## Model Architecture Summary
* **Feature Engineering**: Descriptor selection only for Gaussian NB model. both have cleanning (remove zero variance cols, extreme cols)
* **Ensemble Learning**: ensemble Gaussian NB with selected Descriptors and Bernoulli NB with all cleaned Morgan Fingerprints.
* **Class Balancing**: Applies a `class_weight='balanced'` parameter to natively counteract the severe 75/25 class imbalance of the dataset during splitting.

## How to Use

All outputs will be generated directly.
And each ouputs have corresponding text description besides.

## Results
Reported in the report
