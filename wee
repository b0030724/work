from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt


# Split the data into a training set and a testing set
X = df[['VADER Compound', 'AFINN Score', 'TextBlob Polarity']]
y = labeled_data['GroundTruth']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize machine learning models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}


# Initialise a DataFrame to store the results
results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Macro F1', 'Micro F1', 'Precision', 'Recall', 'F1'])


# Train and evaluate machine learning models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')  # Calculate macro-average F1 score
    micro_f1 = f1_score(y_test, y_pred, average='micro')  # Calculate micro-average F1 score
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True, zero_division=1)


    # Extract precision, recall, and F1 from the report
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']


    # Add the results to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame({'Model': [model_name], 'Accuracy': [accuracy], 'Macro F1': [macro_f1], 'Micro F1': [micro_f1], 'Precision': [precision], 'Recall': [recall], 'F1': [f1]})], ignore_index=True)
    report_df = pd.DataFrame(report).transpose()  # Convert classification report to a DataFrame
    # Print the results DataFrame
    print(report_df)
    print(results_df)


# Calculate class distribution
class_distribution = labeled_data['Sentiment'].value_counts()


# Visualise class distribution
plt.bar(class_distribution.index, class_distribution.values)
plt.xlabel('Sentiment Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()


# Check for class imbalance
if len(class_distribution) > 1:
    imbalance_ratio = class_distribution.min() / class_distribution.max()
    if imbalance_ratio < 0.2:  # You can adjust this threshold as needed
        print("Class imbalance detected. Consider addressing it.")