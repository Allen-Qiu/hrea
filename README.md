# Extracting Hierarchical Relationships of Aspects from Reviews Using Self-supervised Learning
This website shows the amazon datasets and related codes. Others will be shared after the paper is accepted.

dependencies: Pytorch, cudn, sci-kit learn, NLTK 

Cite: Jiangtao Qiu, Ling Lin, Siyu Wang, Extracting hierarchical relationships of aspects from reviews, Applied Soft Computing, Volume 178, 2025, 113335,

## Steps
To train the model, please follow these steps:

(1) unzip the file amazon-review-100k.rar

(2) run the file hrea_amazon_datapreprocess.py to process dataset.

(3) run the file hrea_amazon_encoder_train.py to train a text autoencoder.

(4) run the file hrea_amazon_encoder_visual.ipynb to see the visulized reviews 

(5) run the file hrea_amazon_level1_train.py to train the first student model

(6) run the file hrea_amazon_level1_explore.py to explore the aspect-groups in first level

(7) run the file hrea_amazon_level2_train.py to train the second student model

(8) run the file hrea_amazon_level2_explore.py to explore the aspect-groups in second level

## Datasets

(1) amazon-aspects-category-distribution.txt includes the category-distribution of aspect terms

(2) amazon-embeds.txt includes the words embeddings

(3) amazon-aspects.txt includes the aspect groups

(4) amazon-aspect-group-category-level1.txt includes aspect groups in the first level with categories

(5) amazon-aspect-group-category-level2.txt includes aspect groups in the second level with categories

(6) yelp-reviews.json includes reviews and ratings selected from Yelp dataset.

(7) wecr-reviews-10c.json is The Women’s E-Commerce Clothing Reviews 

## Hyper-parameters

The hyperparameters for training the models are listed in Table 1.

![alt text](images/hyperparams.png)

## Hierarchical relations of aspects in the Amazon dataset

This is the hierarchical relationships of aspects extracted from the Amazon dataset with 50 banks in the first level and 10 banks in the second level. The categories of aspect groups are highlight in red. They are inferred by selecting the top two highestscoring categories from their category distributions. For each aspect group, the first five aspect terms are selected for presentation.

![alt text](images/fig6.png)

## Aspects clusters detected by GMM
![alt text](images/gmm.png)

## Aspects clusters detected by LDA
![alt text](images/LDA.png)

## Aspects clusters detected by ABAE
![alt text](images/abae.png)

## Aspects clusters detected by LCC+GBC
![alt text](images/lcc.png)

## Aspects clusters detected by SSCL
![alt text](images/sscl.png)
