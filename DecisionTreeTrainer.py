import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

# Prep Loan.csv
df = pd.read_csv('C:\Users\Zegert\Documents\GitHub\Loan_elegibility_Confirmation_GroupE\Loans.csv')

irrelevant_features = ['Loan_ID']
df.drop(columns=irrelevant_features, inplace=True)

ohe = pd.get_dummies(df, columns=['Married', 'Gender', 'Education', 'Self_Employed', 'Property_Area'])

X = ohe.drop('Loan_Status', axis=1)
y = ohe['Loan_Status']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 2))

# Decision Tree
depth_hyperparams = range(1, 16)
training_acc = []
validation_acc = []

for d in depth_hyperparams:
    model_dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    model_dt.fit(X_train, y_train)
    training_acc.append(model_dt.score(X_train, y_train))
    validation_acc.append(model_dt.score(X_val, y_val))

tune_data = pd.DataFrame({'Training': training_acc, 'Validation': validation_acc}, index=depth_hyperparams)
fig = px.line(data_frame=tune_data, x=depth_hyperparams, y=['Training', 'Validation'], title="Training & Validation Curves (Decision Tree Model)")
fig.update_layout(xaxis_title="Maximum Depth", yaxis_title="Accuracy Score")
fig.show()

# Evaluate Decision Tree
final_model_dt = DecisionTreeClassifier(max_depth=6, random_state=42)
final_model_dt.fit(X_train, y_train)
y_val_pred = final_model_dt.predict(X_val)
print(f'Final model accuracy: {accuracy_score(y_val, y_val_pred)}')

feature_names = X.columns
plt.figure(figsize=(18, 12))
plot_tree(decision_tree=final_model_dt, filled=True, max_depth=2, feature_names=feature_names, class_names=True)
plt.axis('off')
plt.show()
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def train_decision_tree(C:\Users\Zegert\Documents\GitHub\Loan_elegibility_Confirmation_GroupE\Loans.csv):
    df = pd.read_csv(file_path)
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    clf = DecisionTreeClassifier(random_state=1)
    clf.fit(X_train, y_train)

    # Predict the loan status for the test data
    y_pred = clf.predict(X_test)

    print("DEcision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred))

    fig = plt.figure(figsize=(15,10))
    _ = tree.plot_tree(clf, 
                       feature_names=X.columns,  
                       class_names=['Loan Approved', 'Loan Not Approved'],
                       filled=True)

    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

train_decision_tree(r'C:\Users\Zegert\Documents\GitHub\Loan_elegibility_Confirmation_GroupE\Loans.csv')
'''