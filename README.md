# CSCI-635-01-Group-3

## Forest Cover Type Prediction

## Abstract

This project aims to predict the forest cover type from cartographic variables only. This repository contains the steps to build, evaluate and interpret multiple machine learning models for this multiclass classification problem

## Developers

#### Adrian Lariani

- **decisionTree.ipynb**: model selection, training, testing, and evaluating
- **knn.ipynb**: model selection, training, testing, and evaluating
- **neuralNetwork.ipynb**: some model exploration
- **bestModels.ipynb**: Model Metrics sections
- **dataAnalysis.ipynb**: Data exploration and graphing

#### Aidan Mayes Poduslo

- **neuralNetwork.ipynb**: model selection, training, testing, and evaluating
- **bestModels.ipynb**: Models SHAP Interpretibility sections
- **dataAnalysis.ipynb**: Data balancing
- **utils.py**: utility functions including preprocessing, shap, and more

## Repository Structure

````
├── code/                           # Contains all source code, scripts, and Jupyter notebooks
│   └── decisionTree.ipynb          # Notebook with code on training process of the decision tree model
│   └── knn.ipynb                   # Notebook with code on training process of the knn model
│   └── neuralNetwork.ipynb         # Notebook with code on training process of the neuralNetwork model
│   └── bestModels.ipynb            # Notebook with code on evaluating best of the 3 models with interpretability through SHAP
│   └── dataAnalysis.ipynb          # Notebook with code on exploring the dataset and sampling options
│   └── utils.py                    # File containing utility functions used in most notebooks
├── data/                           # Contains the dataset
│   └── best_decision_tree.joblib   # Best Performing Decision Tree trained, trained in decisionTree.ipynb
│   └── best_knn.joblib             # Best Performing KNN trained, trained in knn.ipynb
├── resources/                      # Contains presentation slides, papers, and other references
│   └── presentation.pdf
├── README.md                       # Project overview and instructions
└── requirements.txt                # Lists all required Python packages```
````

## How to Run the Project

##### 1. Clone Repository

`git clone https://github.com/ahm5348/CSCI-635-01-Group-3.git`
`cd CSCI-635-01-Group-3`

##### 2. Setup Virtual Environment

```
python -m venv venv
.\venv\Scripts\activate
```

##### 3. Install Requirements

```
pip install -r requirements.txt
```

##### 4. Download dataset or use import in notebooks

https://archive.ics.uci.edu/dataset/31/covertype

##### 5. Run notebooks

1. Launch jupyter notebook or run in VSCode or some other editor

```
 jupyter notebook
```

2. Navigate to code/ directory and open on of the .ipynb files
3. Run the cells in a .ipynb file sequentially to perform data preprocessing, training of the models, evaluation metrics, and visualizations of the results
