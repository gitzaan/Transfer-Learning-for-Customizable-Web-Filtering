# Customized URL Filtering - Feature Engineering and Transfer Learning

##Authors 

- [Pronoy Kundu](https://github.com/Pronoy513)
- [Fowzaan Rasheed](https://github.com/gitzaan/)
- [Syed Sahil](https://www.youtube.com/watch?v=dQw4w9WgXcQ)
- [Sachin Saravana](https://www.youtube.com/watch?v=dQw4w9WgXcQ))


## Introduction

This repository contains code and data for a research project focused on creating a customized URL Classification using transfer learning techniques. The project aims to develop a machine learning model capable of effectively blocking access to malicious and inappropriate websites, tailored to the specific requirements of different organizations.

## Data Preparation

We start by preparing and exploring the dataset containing features related to URLs. The dataset is loaded, and several preprocessing steps are performed, including data cleaning and feature selection.

## Feature Analysis

We conduct an analysis of the dataset to understand the variation of features across different types of URLs (Benign, Defacement, Malware, Phishing, and Spam). A plot is generated to visualize how different features vary for each URL type.


## Transfer Learning and Neural Network

The project involves training and testing a neural network model using features extracted from the dataset. We use transfer learning techniques to adapt the model for customized URL classification. The model architecture, including embedding layers and a bidirectional GRU, is defined and compiled.

## Getting Started

To replicate this research and explore the code and data, follow these steps:

1. Clone this repository: `git clone [repository_url]`
2. Install the required Python libraries: `pip install -r requirements.txt`
3. Run the Jupyter Notebook or Python script to perform feature engineering and transfer learning.

## Requirements

- Python 3.x
- Libraries mentioned in the `requirements.txt` file

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



