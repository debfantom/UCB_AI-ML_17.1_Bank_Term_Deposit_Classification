# UCB_AI-ML_17.1_Bank_Term_Deposit_Classification

This repository contains:
- **notebook.ipynb**: Jupyter Notebook containing exploratory data analysis, feature engineering, model comparison, and tuning.
- **data/**: Folder containing the dataset used in this project.
- **images/**: Folder with generated plots and visualizations.

---

## 📌 Project Description

This project explores a binary classification problem using a bank marketing dataset. The task is to predict whether a client will subscribe to a term deposit (`y = 'yes'`) based on demographic, campaign, and economic features. I evaluated multiple classifiers and focused on optimizing recall and ROC AUC for the minority class.

---

## ❓ Problem

Only about **12% of clients** subscribe to a term deposit, making this an imbalanced classification problem. While a model predicting "no" for every case would score ~88% accuracy, it would fail completely at identifying the successful outcomes that are actually important.

Instead of optimizing for accuracy, I focused on:
- **Recall** (to maximize captured "yes" cases)
- **ROC AUC** (to assess overall ranking quality)

A meaningful model should achieve **recall > 0.30** and **ROC AUC > 0.70** to outperform random guessing.
After some initial exploratory data analysis (EDA), I made the following hypotheses to guide my analysis.  My hope was to identify areas of opportunity to increase the program's success (as measured by term deposit offer acceptance) through my understanding of what factors were correlated to a customer's increased probability of accepting an offered term deposit.

---

#### Hypotheses:

**Primary Hypothesis:**
Customer coupon acceptance rates are strongly influenced by prior patronage habits (visits per month) specific to the type of establishment (e.g., carry out, coffee house, or cheap restaurant). Customers with a higher historical frequency of visiting such establishments are more likely to accept corresponding coupons.

**Secondary Hypotheses:**

Coupon Expiration Dynamics:

Coupons with a shorter expiration period (e.g., 2 hours) may lead to higher immediate usage, while longer expiration periods (e.g., 1 day) may encourage acceptance for future use. Since success for our use case includes acceptance for later usage, offering the longer expiration in all cases may help boost acceptance rates.  We might also be open to testing longer experation periods.

---

## 📊 Visualizations

During EDA, I explored relationships between features and the likelihood of term deposit subscription.

Key insights:
- Subscription rate increases steadily with age, with a notable jump at 60+.
- Clients with university degrees had higher subscription rates than others.
- Previous campaign success and prior contact were highly correlated with positive outcomes.

Plots are available in the `images/` directory and embedded in the notebook.

---

## 🧠 Models Compared

I implemented and compared four classifiers:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)

Each model was evaluated using **recall** and **ROC AUC**, both before and after feature updates and tuning.

---

## 🔬 Evaluation Metrics

All models were assessed using:
- **Recall** on the positive class (`y_binary == True`)
- **ROC AUC** to measure global separability
- **Training time** to measure scalability and efficiency

---

## 📈 Model Performance Summary

### Baseline vs. Feature Engineering Updates vs. with Hyperparameter Tuning

| Model                    | Train Time Baseline | Train Time Features | Train Time w/Tuning | Recall (yes) Baseline | Recall (yes) Features | Recall (yes) w/Tuning | ROC AUC Baseline | ROC AUC Features | ROC AUC w/Tuning |
|--------------------------|------------------------|------------------------|----------------------------|------------------------|------------------------|-----------------------------|-------------------|------------------|------------------------|
| **Logistic Regression**  | 0.0979 s               | 0.0692 s               | 0.7601 s                   | 0.1789                 | 0.1843                 | 0.1853                      | 0.7770            | 0.7777           | 0.7778                 |
| **K–Nearest Neighbors**  | 0.0256 s               | 0.0125 s               | 3.8303 s                   | 0.2769                 | 0.3082                 | 0.3006                      | 0.7193            | 0.7485           | 0.7137                 |
| **Decision Tree**        | 0.0973 s               | 0.0296 s               | 1.0707 s                   | 0.3265                 | 0.2554                 | 0.2597                      | 0.6204            | 0.6457           | 0.7338                 |
| **Support Vector Machine** | 63.3919 s            | 22.2335 s              | 268.667 s                  | 0.2037                 | 0.2144                 | 0.2155                      | 0.6788            | 0.6786           | 0.7085                 |

---

### Summary

- Feature engineering significantly reduced training time and often improved both recall and ROC AUC.
- Tuning hyperparameters boosted ROC AUC for all models, especially the Decision Tree (from 0.6457 to 0.7338).
- K-Nearest Neighbors maintained the highest recall across all phases, although its AUC decreased slightly with tuning.
- Logistic Regression remained the most consistent and interpretable model, with top AUC and stable recall.
- Support Vector Machine showed modest performance gains, but incurred a significant increase in training time due to the tuning process.

---

---

## 🔄 Next Steps

Future improvements could include:
- Testing ensemble methods (e.g., Random Forest, XGBoost)
- Addressing class imbalance using SMOTE or cost-sensitive learning
- Refining threshold decisions to improve the balance between recall and precision
- Adding time-aware modeling for macroeconomic trends

---

## 🧪 Environment & Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn

To install dependencies:

```bash
pip install -r requirements.txt

