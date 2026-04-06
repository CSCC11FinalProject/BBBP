# CSCC11 Final Project: Predicting Blood-Brain Barrier Permeability (BBBP)

**Group Members:** Max, Michael, Fan, Jerry

## Abstract
The prediction of blood-brain barrier permeability (BBBP) is a significant challenge in neuropharmacology, as it determines the effectiveness of therapeutic agents targeting the central nervous system. This project benchmarks four predictive models\textemdash KNN, Ensemble Naive Bayes, Random Forest, and Message-Passing Neural Network\textemdash using an imbalanced dataset of $\approx 2050$ organic molecules, evaluating performance based on AUROC and F1-Score. While the Random Forest achieved the highest AUROC ($0.911$) and tied for highest recall ($0.957$), the MPNN demonstrated superior precision ($0.938$) and the highest F1-Score ($0.928$), with near-identical AUROC ($0.910$). As the high cost of false-positive predictions in drug discovery directly leads to wasted resources in clinical trials, the MPNN was selected as the optimal model. By processing molecules as graphs, the MPNN can identify specific atomic relationships that traditional, descriptor-based models often overlook, making it a more precise tool for identifying feasible neurotherapeutics.

## Repository Structure
The project is organized into model-specific directories, each containing its own scripts, plots, and supplementary documentation:
* `dataset/`: Contains the raw `BBBP.csv` dataset, along with data preprocessing, exploratory data analysis (EDA), and RDKit feature extraction scripts (`process.py`, `visualize.py`).
* `KNN/`: K-Nearest Neighbors baseline model implementation and tuning.
* `NaiveBayes/`: Dual-component Ensemble Naive Bayes model (Gaussian for continuous descriptors, Bernoulli for binary fingerprints).
* `RandomForest/`: Random Forest classifier implementation utilizing 2048-bit Morgan Fingerprints.
* `MPNN/`: Message-Passing Neural Network featuring graph convolutions (`NNConv`), GRU state updates, and self-attention pooling. 

Each folder contains the scripts necessary to tune, train and evaluate the model. 


## Installation and Setup

To get started, create a conda env (Python 3.10+) and run `pip install -r requirements.txt`.

Each model can be executed from within its respective directory. For example, to train and evaluate the MPNN:
```bash
    cd MPNN
    python train.py
    python evaluate.py
```

*Refer to the `readme.md` files located inside the model subfolders for more details*

## Final Results
| Model | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- |
| **KNN Baseline** | 0.855 | 0.957 | 0.903 | 0.850 |
| **Ensemble Naive Bayes** | 0.893 | 0.932 | 0.912 | 0.873 |
| **Random Forest** | 0.889 | 0.957 | 0.922 | 0.911 |
| **MPNN** | 0.938 | 0.919 | 0.928 | 0.910 |

### Conclusion
The **MPNN** is selected as the final model for predicting BBB permeability. Evaluating these models for real-world pharmacology requires prioritizing precision over recall. In early-stage drug discovery, a false positive is highly detrimental; because physical testing resources are severely constrained, incorrectly predicting that a compound will cross the blood-brain barrier wastes resources in clinical trials and expensive experimental testing. Thus, although the Random Forest achieved a similar F1-score and AUC-ROC, and higher recall, and is less computationally complex in training, we still concluded that the MPNN is the better model, as it achieved higher precision, which is critical in real-world pharmacological applications.