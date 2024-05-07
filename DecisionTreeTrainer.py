import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def train_decision_tree(file_path):
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

train_decision_tree(r'C:\Users\delig\Documents\Python Scripts\machine_learning_course\Loans.csv')