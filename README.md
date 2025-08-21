# 🌾 Crop Classification with Machine Learning

This project applies a variety of machine learning techniques to classify **Oats vs. Other crops** using the **WinnData dataset**.  
The goal is to explore classical and modern models, evaluate their performance, and identify the most effective classifier under class imbalance.

---

## 🚀 Project Overview
- Preprocessed and cleaned the dataset, including:
  - Removing near-zero variance attributes.
  - Handling missing values and balancing classes via up/undersampling.
- Trained and evaluated multiple models:
  - **Decision Tree, Naïve Bayes, Bagging, Boosting, Random Forest, LASSO Regression, ANN, and XGBoost**.
- Benchmarked models using:
  - **Confusion Matrix, Accuracy, Precision, Recall, F1-score, ROC-AUC**.
- Explored **feature importance** for interpretability across models.

---

## 📊 Key Results
- **Best Model**: Balanced Random Forest with tuned `mtry` achieved the highest AUC and recall for the minority "Oats" class.
- **XGBoost** and **ANN** further improved classification with oversampling and feature scaling.
- Demonstrated the impact of data balancing techniques on model performance.

---

## 🧰 Tech Stack
- **Languages**: R  
- **Libraries**: `caret`, `tree`, `randomForest`, `adabag`, `glmnet`, `neuralnet`, `pROC`, `xgboost`  
- **Techniques**:  
  - Data preprocessing & class balancing (up/undersampling)  
  - Feature selection (LASSO, Random Forest importance)  
  - Ensemble methods (Bagging, Boosting, Random Forest, XGBoost)  
  - Neural networks  

---

## 📈 Visualisations
- Confusion matrices for each model.
- ROC-AUC curves comparing classifiers.
- Feature importance plots for interpretability.

---

## 🔑 Learning Outcomes
- Hands-on experience in **applied machine learning and model benchmarking**.
- Deeper understanding of **imbalanced classification problems**.
- Practical exposure to **ensemble learning, regularisation, and neural networks** in R.


---

## 📝 Author
👤 **Brennen Chong**  
Penultimate-year Computer Science student @ Monash University | Data Science & Software Engineering Enthusiast  
