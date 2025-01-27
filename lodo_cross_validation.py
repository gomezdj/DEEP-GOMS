import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
from glob import glob

# Load datasets
files = glob('datasets/*.csv')
data_frames = [pd.read_csv(file) for file in files]
cohorts = [os.path.basename(file).split('.')[0] for file in files]

# Combine datasets for a single cross-validation evaluation
all_data = pd.concat(data_frames, keys=cohorts).reset_index(level=1, drop=True)

def prepare_data_for_rf(data):
    # One-hot encode categorical variables, handle missing data if any
    data = pd.get_dummies(data, columns=['Sex'])
    data.fillna(data.mean(), inplace=True)
    labels = data['Response'].map({'Responder': 1, 'Non-responder': 0})
    features = data.drop(columns=['SampleID', 'Response'])
    return features, labels

def evaluate_auc(model, X, y):
    y_pred = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)
    return auc

# LODO Evaluation
results = {}
for cohort in cohorts:
    print(f"Evaluating leave-out for {cohort}")
    train_data = all_data[all_data.index != cohort]
    test_data = all_data[all_data.index == cohort]
    
    X_train, y_train = prepare_data_for_rf(train_data)
    X_test, y_test = prepare_data_for_rf(test_data)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    auc = evaluate_auc(rf_model, X_test, y_test)
    results[cohort] = auc
    print(f"AUC for {cohort}: {auc:.2f}")

# Single cross-validation evaluation
X_all, y_all = prepare_data_for_rf(all_data)
rf_model_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_all.fit(X_all, y_all)

single_auc = evaluate_auc(rf_model_all, X_all, y_all)
print(f"AUC for single cross-validation evaluation: {single_auc:.2f}")

# Plot ROC Curve for the combined model
y_pred = rf_model_all.predict_proba(X_all)[:, 1]
fpr, tpr, _ = roc_curve(y_all, y_pred)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {single_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("results/roc_curve_combined.png")
plt.show()

# Save results
results_df = pd.DataFrame(list(results.items()), columns=['Cohort', 'AUC'])
results_df.to_csv('results/lodo_auc_results.csv', index=False)
print("LODO AUC results saved to 'results/lodo_auc_results.csv'")
