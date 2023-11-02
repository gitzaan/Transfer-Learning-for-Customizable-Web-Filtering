# Transfer Learning for Customized URL Filtering - URL Classification with CNN/LSTM Hybrid Model 

## Authors 

- [Pronoy Kundu](https://github.com/Pronoy513)
- [Fowzaan Rasheed](https://github.com/gitzaan/)
- [Syed Sahil](https://github.com/syed-sahil-100)
- [Sachin Saravana](https://github.com/SachinSarv1473)


## Overview 

This repository contains a URL classification system using Neural Networks. It aims to classify various URLs into different categories, such as Benign, Malware, Phishing, Spam, and Defacement. The project is based on an original implementation provided in the [Medium Post](https://medium.com/nerd-for-tech/url-feature-engineering-and-classification-66c0512fb34d) by the authors, Aaditya Jain, Anirudh Bhaskar, Srikanth, and Rohith Ramakrishnan. We have made modifications to the feature extraction and the model used for classification.

## Set-Up

### Pre-requisites

Before running this project, make sure you have the following installed:

- [conda](https://repo.anaconda.com/)
- [git](https://git-scm.com/)

### Installation

Clone the repository and create a Python environment with the required packages:
```

git clone https://github.com/gitzaan/Transfer-Learning-for-Customizable-Web-Filtering
cd Transfer-Learning-for-Customizable-Web-Filtering
conda create -n pyenv python=3.8.5
conda activate pyenv
pip install -r requirements.txt
```

## Feature Extraction

```
cd scripts/
python UrlFeaturizer.py

```

The features extracted are explained and visualized in the DataProcessing.ipynb notebook.


## Data Description via Extracted Features 

The project extracts various features from URLs, which are categorized into different groups. These features include characteristics of the URL string, domain features, and page features. For a detailed list of features, please refer to the original [Medium Post](https://medium.com/nerd-for-tech/url-feature-engineering-and-classification-66c0512fb34d.)

## Model

In our modified version of the project, we use a Convolutional Neural Network (CNN) with LSTM layers for classification. The architecture includes convolutional layers, batch normalization, max-pooling, dropout layers, and fully connected layers. The model is trained with an Adam optimizer.

You can train the model using :
```
cd scripts/
python modelTrain.py

```
## Model Evaluation

After training the model, we evaluate its performance using metrics like accuracy and generate a classification report. The report provides detailed information on the model's classification performance for different categories.

## Making Predictions

To make predictions on a new URL, you need to follow these steps:

1. Load the LabelEncoder and StandardScaler from the saved files.
2. Load the pre-trained model.
3. Featurize the URL and prepare the features for prediction.
4. Standardize the features and reshape them to match the model's input shape.
5. Make predictions and convert numerical predictions to class labels.

- You can make predictions by running the following
 ```
 cd scripts/
python predict_args.py -i <url>
```
## License
This project is licensed under the MIT License - see the LICENSE file for details.
