## **Automatically classifying native texts in Icelandic based on their CEFR-levels**

This project analyzes the presence of thirty grammatical features present in two data sets where Icelandic texts have been labelled according to their level in The Common European Framework of Reference for Languages (CEFR). The idea is to facilitate feature engineering for automatic classification of texts written by native speakers which in turn can be used for teaching and evaluation of Icelandic as a second language. Additionally, a baseline classification model is proposed.

**[IGC_texts_unreviewed](https://github.com/steinunnfridriks/CEFR_firstlanguage_classification/tree/main/IGC_texts_unreviewed)** is a collection of txt files collected from the Icelandic Gigaword Corpus. They have been sorted into separate directories based on their proposed CEFR-level (but the categorization has not been reviewed by an expert in Icelandic as a second language). The levels are imbalanced in terms of the number of texts. No texts were found on the A1 level. The A2 level contains 8 texts, the B1 level contains 23 texts, the B2 level contains 76 texts, the C1 level contains 166 texts and the C2 level contains 82 texts. 

**[data_process.py](https://github.com/steinunnfridriks/CEFR_firstlanguage_classification/blob/main/data_process.py)** is a simple preprocessing script that goes through all txt documents present in the previously mentioned directories and outputs a csv file with columns representing the texts, their location in the Icelandic Gigaword Corpus and their label as proposed in the data. The accompanying file [IGC_texts.csv](https://github.com/steinunnfridriks/CEFR_firstlanguage_classification/blob/main/IGC_texts.csv) was created using this script. 

**[level_analyser.py](https://github.com/steinunnfridriks/CEFR_firstlanguage_classification/blob/main/data_process.py)** takes a csv file as an input. The csv file should have at least two columns where one represents the text and the other represents the CEFR-level of the text. This script analyzes the presence of various syntactic and grammatical features and calculates their average percentage in texts from each level. The analysis is independent of model structure. It should provide insight into the linguistic properties of each level and thus facilitate feature engineering needed to train a classifier.

**[gradientboosting.py](https://github.com/steinunnfridriks/CEFR_firstlanguage_classification/blob/main/gradientboosting.py)** uses the features analyzed in the previous script to build feature vectors representing the texts in the data. The script assumes that the format of the data is as described above, containing at minimum the colums _text_ and _cefrlevel_. The classifier trained using this script is a gradient boosting model whose performance is measured using the average accuracy on 10-fold cross validation.

**[The accompanying report](https://github.com/steinunnfridriks/CEFR_firstlanguage_classification/blob/main/cefrsk%C3%BDrsla.pdf)** (in Icelandic) explains the process in more detail and includes graphs indicating the importance of the evaluated features as measured in the previously mentioned data as well as data collected by Kolfinna Jónatansdóttir in the fall of 2023. The latter is not included in this repository as a part of the data has not been cleared for publishing. 

This project was funded by [Styrktarsjóður Áslaugar Hafliðadóttur](https://sjodir.hi.is/styrktarsjodur_aslaugar_haflidadottur).
