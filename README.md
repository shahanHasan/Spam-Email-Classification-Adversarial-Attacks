![Python](https://img.shields.io/badge/Python-Version%20%3A%203.0-yellowgreen)
![Conda](https://img.shields.io/badge/conda%20-4.10.3-blackyellow)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Naereen/badges)
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)


# Spam Email Classification | Adversarial Attacks-

## *1. A comparative study of Machine learning classifiers and deep neural network learning algorithms applied to the problem of Spam emails* ##

## *2. Exploration of Adversarial attacks on spam email classification learning algorithms* ##

**The Methodology/Pipeline of the system is illustrated below :**

* Data Preprocessing
* Model Training
* Model Testing and Evaluation

![Methodology](Methodology.jpg)

### Machine Learning Classifiers :
1. Multinomial Naive Bayes
2. Logistic Regression
3. Support Vector Machines ( linear and Radial basis function )
4. K-Nearest Neighbours
5. K-means Clustering
6. Random Forest 
7. Gradient Boosting 
8. XGBoost 
9. Decision Tree 

### Deep Learning Algorithms : 
1. DNN ( with word Embeddings )
2. RNN ( with word Embeddings )
3. CNN ( with word Embeddings )

4. DNN ( with word Pretrained Glove Embeddings )
5. RNN ( with word Pretrained Glove Embeddings )
6. CNN ( with word Pretrained Glove Embeddings )

### Feature Extraction techniques for Machine Learning Classifiers  : 
**tf-idf -> Term Frequency , Inverse Term Frequency**

---

### Feature Extraction techniques for Deep Learning Classifiers     : 
**Word Embeddings ( Trainable ) and ( Non - Trainable : glove )** 

---

### Metrics for Evaluation : 
1. **Accuracy**
2. **f1-Score**
3. **Precision**
4. **Recall**
5. **ROC-AUC**
6. **Error and Loss**

---


# Adversarial Attacks - 
1. **Label Flipping**

---

![Algorithm 1 : Label Flipping](alg-1.png)


2. **Sample Poisoning (Synonym Replacement, Spam / Ham word Injection**

---

### Algorithm 2-1 : Synonym Replacement : 

![Algorithm 2-1 : Augment emails with synonyms in a certain number of records indexed randomly and derived from a threshold](alg-2-1.png)



### Algorithm 2-2 : Spam / Ham word Injection :

![Algorithm 2-2 : Augment emails with Ham or Spam words in a certain number of records indexed randomly and derived from a threshold / HAM or SPAM word Injection](alg-2-2.png)

---

### Algorithm 3 : Addition of new Poisoned Emails algorithm (To do):

![Algorithm 3: Addition of new Poisoned emails algorithm, Make new emails with randomly selected ham and spam words.](alg-3.png)

---

# Adversarial Attacks on Deep Learning Models (To do):
1. **Fast Gradient Method (FGM)**

---

2. **Fast Gradient Sign Method (FGSM)**

---

3. **L2 Projected Gradient Descent (PGD)**

---

4. **Linf Projected Gradient Descent (LinfPGD)**

---


# Adversarial attacks defensive mechanism (To do):

1. **Application of KNN as Defence**

---

![Algorithm 4 : Defence Mechanism](alg-4.png)

# Directory Structure of the repo :

```
.
|-- Architecture_Diagrams
|   |-- ann_1.jpeg
|   |-- ann_glove_1.jpeg
|   |-- ANN_glove_1.jpeg
|   |-- ann_glove_2.jpeg
|   |-- ANN_glove_2.jpeg
|   |-- ANN_word_Embedding.jpeg
|   |-- cnn_1.jpeg
|   |-- cnn_glove_1.jpeg
|   |-- CNN_glove_1.jpeg
|   |-- cnn_glove_2.jpeg
|   |-- CNN_glove_2.jpeg
|   |-- CNN_word_Embedding.jpeg
|   |-- rnn_1.jpeg
|   |-- rnn_glove_1.jpeg
|   |-- RNN_glove_1.jpeg
|   |-- rnn_glove_2.jpeg
|   |-- RNN_glove_2.jpeg
|   `-- RNN_word_Embedding.jpeg
|-- comparison
|   |-- classifiers (All Saved Models)
|   |   |-- Decision_tree.pkl
|   |   |-- Gradient_Boosting.pkl
|   |   |-- KNN.pkl
|   |   |-- Logistic_regression.pkl
|   |   |-- MultinomialNB.pkl
|   |   |-- Random_forest.pkl
|   |   |-- SVM_linear.pkl
|   |   |-- SVM_RBF.pkl
|   |   |-- train_test_tf_idf.pkl
|   |   `-- XGBoost.pkl
|   |-- All_models_and_classifiers.csv
|   |-- Classifier_Metrics_Comparison.csv
|   |-- Classifier_Metrics_Comparison_percentage.csv
|   |-- DNN_glove_1_comparison.csv
|   |-- DNN_glove_2_comparison.csv
|   |-- DNN_Models.csv
|   |-- DNN_Models_percent.csv
|   |-- DNN_Trainable_Embeddings_comparison.csv
|   |-- Gradient_boosting_hyperparameters.csv
|   |-- KNN_hyperparameters.csv
|   |-- Label_flip_Adversarial.csv
|   |-- Metrics_Comparison.csv
|   |-- Spam_Ham_Injection_Adversarial.csv
|   |-- Spam_Ham_Injection_random_Adversarial.csv
|   |-- Synonym_Adversarial.csv
|   `-- Tuned_KNN_GB.csv
|-- Datasets
|   |-- archive
|   |   `-- emails.csv
|   `-- spam_dataset_1
|       `-- emails.csv
|-- EDA
|   |-- LEAST_Word_Counts_0.jpeg
|   |-- LEAST_Word_Counts_10.jpeg
|   |-- LEAST_Word_Counts_11.jpeg
|   |-- LEAST_Word_Counts_12.jpeg
|   |-- LEAST_Word_Counts_13.jpeg
|   |-- LEAST_Word_Counts_14.jpeg
|   |-- LEAST_Word_Counts_15.jpeg
|   |-- LEAST_Word_Counts_16.jpeg
|   |-- LEAST_Word_Counts_17.jpeg
|   |-- LEAST_Word_Counts_18.jpeg
|   |-- LEAST_Word_Counts_19.jpeg
|   |-- LEAST_Word_Counts_1.jpeg
|   |-- LEAST_Word_Counts_20.jpeg
|   |-- LEAST_Word_Counts_21.jpeg
|   |-- LEAST_Word_Counts_22.jpeg
|   |-- LEAST_Word_Counts_23.jpeg
|   |-- LEAST_Word_Counts_24.jpeg
|   |-- LEAST_Word_Counts_25.jpeg
|   |-- LEAST_Word_Counts_26.jpeg
|   |-- LEAST_Word_Counts_27.jpeg
|   |-- LEAST_Word_Counts_28.jpeg
|   |-- LEAST_Word_Counts_29.jpeg
|   |-- LEAST_Word_Counts_2.jpeg
|   |-- LEAST_Word_Counts_3.jpeg
|   |-- LEAST_Word_Counts_4.jpeg
|   |-- LEAST_Word_Counts_5.jpeg
|   |-- LEAST_Word_Counts_6.jpeg
|   |-- LEAST_Word_Counts_7.jpeg
|   |-- LEAST_Word_Counts_8.jpeg
|   |-- LEAST_Word_Counts_9.jpeg
|   |-- MOST_Word_Counts_0.jpeg
|   |-- MOST_Word_Counts_10.jpeg
|   |-- MOST_Word_Counts_11.jpeg
|   |-- MOST_Word_Counts_12.jpeg
|   |-- MOST_Word_Counts_13.jpeg
|   |-- MOST_Word_Counts_14.jpeg
|   |-- MOST_Word_Counts_15.jpeg
|   |-- MOST_Word_Counts_16.jpeg
|   |-- MOST_Word_Counts_17.jpeg
|   |-- MOST_Word_Counts_18.jpeg
|   |-- MOST_Word_Counts_19.jpeg
|   |-- MOST_Word_Counts_1.jpeg
|   |-- MOST_Word_Counts_20.jpeg
|   |-- MOST_Word_Counts_21.jpeg
|   |-- MOST_Word_Counts_22.jpeg
|   |-- MOST_Word_Counts_23.jpeg
|   |-- MOST_Word_Counts_24.jpeg
|   |-- MOST_Word_Counts_25.jpeg
|   |-- MOST_Word_Counts_26.jpeg
|   |-- MOST_Word_Counts_27.jpeg
|   |-- MOST_Word_Counts_28.jpeg
|   |-- MOST_Word_Counts_29.jpeg
|   |-- MOST_Word_Counts_2.jpeg
|   |-- MOST_Word_Counts_3.jpeg
|   |-- MOST_Word_Counts_4.jpeg
|   |-- MOST_Word_Counts_5.jpeg
|   |-- MOST_Word_Counts_6.jpeg
|   |-- MOST_Word_Counts_7.jpeg
|   |-- MOST_Word_Counts_8.jpeg
|   `-- MOST_Word_Counts_9.jpeg
|-- glove
|   |-- 6471382cdd837544bf3ac72497a38715e845897d265b2b424b4761832009c837
|   |   |-- glove.6B.100d.txt
|   |   |-- glove.6B.200d.txt
|   |   |-- glove.6B.300d.txt
|   |   `-- glove.6B.50d.txt
|   |-- 357baac33090f645e71e253b3295ee1b767c98a0336e9a1d99c77e9e33b43c4a.zip
|   |-- 6471382cdd837544bf3ac72497a38715e845897d265b2b424b4761832009c837.zip
|   `-- glove.42B.300d.txt
|-- heatmaps
|   |-- ANN_glove_2.jpeg
|   |-- ANN_glove.jpeg
|   |-- ANN.jpeg
|   |-- CNN_glove_2.jpeg
|   |-- CNN_glove.jpeg
|   |-- CNN.jpeg
|   |-- Descision_Tree.jpeg
|   |-- Gradient_boosting.jpeg
|   |-- KNN.jpeg
|   |-- Linear_SVC.jpeg
|   |-- Logistic_Regression.jpeg
|   |-- Naive_Bayes.jpeg
|   |-- Random_Forest.jpeg
|   |-- RNN_glove_2.jpeg
|   |-- RNN_glove.jpeg
|   |-- RNN.jpeg
|   |-- SVC.jpeg
|   |-- XGBoost.jpeg
|   `-- XGBoost_tuned.jpeg
|-- Model
|   |-- ANN_glove_1.h5
|   |-- ANN_glove_2.h5
|   |-- ANN.h5
|   |-- CNN_glove_1.h5
|   |-- CNN_glove_2.h5
|   |-- CNN.h5
|   |-- model.zip
|   |-- RNN_glove_1.h5
|   |-- RNN_glove_2.h5
|   `-- RNN.h5
|-- __pycache__
|   `-- utils.cpython-38.pyc
|-- Visuals
|   |-- Acc-Loss
|   |   |-- ann_accuracy_loss.jpeg
|   |   |-- ann_glove_accuracy_loss_2.jpeg
|   |   |-- ann_glove_accuracy_loss.jpeg
|   |   |-- cnn_accuracy_loss.jpeg
|   |   |-- cnn_glove_accuracy_loss_2.jpeg
|   |   |-- cnn_glove_accuracy_loss.jpeg
|   |   |-- rnn_accuracy_loss.jpeg
|   |   |-- rnn_glove_accuracy_loss_2.jpeg
|   |   `-- rnn_glove_accuracy_loss.jpeg
|   |-- AU-ROC
|   |   |-- AUC_ANN_GLOVE_2.jpeg
|   |   |-- AUC_ANN_GLOVE.jpeg
|   |   |-- AUC_ANN.jpeg
|   |   |-- AUC_CNN_GLOVE_2.jpeg
|   |   |-- AUC_CNN_GLOVE.jpeg
|   |   |-- AUC_Descision_Tree.jpeg
|   |   |-- AUC_Gradient_Boosting.jpeg
|   |   |-- AUC_KNN.jpeg
|   |   |-- AUC_KNN_tuned.jpeg
|   |   |-- AUC_Logistic_regression.jpeg
|   |   |-- AUC_NB.jpeg
|   |   |-- AUC_NB_tuned.jpeg
|   |   |-- AUC_Random_Forest.jpeg
|   |   |-- AUC_RNN_GLOVE_2.jpeg
|   |   |-- AUC_RNN_GLOVE.jpeg
|   |   |-- AUC_RNN.jpeg
|   |   |-- AUC_SVC.jpeg
|   |   |-- AUC_SVM_Linear.jpeg
|   |   |-- AUC_XGBoost.jpeg
|   |   `-- AUC_XGBoost_tuned.jpeg
|   |-- T-SNE
|   |   |-- ann_Embeddings_1.jpeg
|   |   |-- ann_glove_Embeddings_1.jpeg
|   |   |-- ann_glove_Embeddings_2.jpeg
|   |   |-- cnn_Embeddings_1.jpeg
|   |   |-- cnn_glove_Embeddings_1.jpeg
|   |   |-- cnn_glove_Embeddings_2.jpeg
|   |   |-- rnn_Embeddings_1.jpeg
|   |   |-- rnn_glove_Embeddings_1.jpeg
|   |   `-- rnn_glove_Embeddings_2.jpeg
|   |-- Ham_Overall_Frequency_Distribution.jpeg
|   |-- ham_vs_spam.jpeg
|   |-- modelComparison.png
|   |-- Overall_Frequency_Distribution.jpeg
|   |-- Spam_Overall_Frequency_Distribution.jpeg
|   |-- wordcloud_ham.jpeg
|   |-- wordcloud_overall.jpeg
|   `-- wordcloud_spam.jpeg
|-- alg-1.png
|-- alg-2-1.png
|-- alg-2-2.png
|-- alg-3.png
|-- alg-4.png
|-- Augmented_emails.csv
|-- Comparison of Models | Adversarial Attacks data Preparation.ipynb
|-- emails.csv
|-- glove.6B.100d.txt
|-- Methodology.jpg
|-- Plot Model Architectures.ipynb
|-- README.md
|-- SPAM Email Classification .ipynb
|-- Spam_Email_Classification_with_ANN,_RNN,_CNN_with_pretrained_glove_Word_embeddings_1_.ipynb
|-- Spam_Email_Classification_with_ANN,_RNN,_CNN_with_pretrained_glove_Word_embeddings_2.ipynb
|-- Spam_Email_Classification_with_ANN,_RNN,_CNN_with_word_embeddings_respectively.ipynb
|-- tfidf.csv
`-- utils.py
```
