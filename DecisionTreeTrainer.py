import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import plotly.express as px
import matplotlib.pyplot as plt

# Prep Loan.csv
df = pd.read_csv('Loans updated2.csv')

df.drop(columns=['Loan_ID'], inplace=True)

ohe = pd.get_dummies(df, columns=['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                                  'Credit_History', 'Married', 'Loan_Status', 'Education',
                                  'Self_Employed', 'Gender', 'Property_Area'])

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

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
#fig.show()

# Evaluate Decision Tree
final_model_dt = DecisionTreeClassifier(max_depth=6, random_state=42)
final_model_dt.fit(X_train, y_train)
y_val_pred = final_model_dt.predict(X_val)
print(f'Final model accuracy: {accuracy_score(y_val, y_val_pred)}')

feature_names = X.columns
plt.figure(figsize=(18, 12))
plot_tree(decision_tree=final_model_dt, filled=True, max_depth=2, feature_names=feature_names, class_names=True)
plt.axis('off')
plt.savefig('decisiontree.png')

def classify_new_client(model, client_details):
    # Prepare data 
    client_df = pd.DataFrame(client_details, index=[0])
    # Make prediction
    prediction = model.predict(client_df)
    return prediction[0]

# Return the final model
final_model_dt = DecisionTreeClassifier(max_depth=6, random_state=42)
final_model_dt.fit(X_train, y_train)