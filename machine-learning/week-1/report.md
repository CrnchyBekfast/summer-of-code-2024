The colab file is here: https://colab.research.google.com/drive/1ThgUaWka1KEr9kzfBBU9W4Sge6aplfRg?usp=sharing

The api python file is in the DSoC folder

I followed the workflow from Google MLCC, and did copy-paste some code from the internet and LLMs here and there, adapting it all to my problem and my dataset

The dataset is here: https://www.kaggle.com/c/ieee-fraud-detection/data

The dataset took a little too long to upload (I couldn't wait more than like two minutes and the bar wasn't moving, so I mounted my google drive to it and imported the dataset from there

I chose the dataset because it seemed to have comparatively a LOT of un-anonymized non-personally identifying data, others were either too simple, or anonymized.

The logistic regression model doesn't seem to work at all with ROC AUC Score: 0.7427, Precision-Recall AUC Score: 0.4973
The Random Forest model is very very good, ROC AUC Score: 0.9707, Precision-Recall AUC Score: 0.9219
The XGBoost model took A LOT OF TUNING, but turned out very decent, with ROC AUC Score: 0.9560, Precision-Recall AUC Score: 0.8784
The Neural Network also doesn't seem to work at all for some reason and I couldn't for the life of me get it to work even half well, it would either overfit or be no better than chance at detecting fraudulent transactions,
it had Precision-Recall AUC Score: 0.4986, ROC AUC Score: 0.7548
