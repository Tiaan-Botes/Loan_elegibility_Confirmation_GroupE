import dash
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/Tiaan-Botes/Loan_elegibility_Confirmation_GroupE/c908b07c97658149902c245a4d0a2589061dec44/Loans%20updated2.csv')
df.drop(columns=['Loan_ID'], inplace=True)

#prepare data
df['Dependents'] = df['Dependents'].astype(str).str.replace('+', '')
df['Dependents'] = df['Dependents'].astype(float)
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mean())
df['Dependents'] = df['Dependents'].round(decimals=2)

# Define DASH app
app = dash.Dash(__name__)
app.title = "Decision Tree Classifier - Loan Eligibility"

# Define layout
app.layout = html.Div([
    html.H1("Decision Tree Classifier - Loan Eligibility Dashboard"),
    
    html.Div([
        html.H3("Box Plot of Applicant Income"),
        dcc.Graph(id='boxplot-applicant-income')
    ]),
    
    html.Div([
        html.H3("Distribution Plot of Applicant Income"),
        dcc.Graph(id='distplot-applicant-income')
    ]),
    
    html.Div([
        html.H3("Box Plot of Coapplicant Income"),
        dcc.Graph(id='boxplot-coapplicant-income')
    ]),
    
    html.Div([
        html.H3("Distribution Plot of Coapplicant Income"),
        dcc.Graph(id='distplot-coapplicant-income')
    ]),
    
    html.Div([
        html.H3("Box Plot of Loan Amount"),
        dcc.Graph(id='boxplot-loan-amount')
    ]),
    
    html.Div([
        html.H3("Distribution Plot of Loan Amount"),
        dcc.Graph(id='distplot-loan-amount')
    ]),
    
    html.Div([
        html.H3("Decision Tree Plot"),
        html.Img(src='decisiontree.png', style={'width':'100%'})
    ]),
])

# Define callbacks
@app.callback(
    Output('boxplot-applicant-income', 'figure'),
    Output('distplot-applicant-income', 'figure'),
    Output('boxplot-coapplicant-income', 'figure'),
    Output('distplot-coapplicant-income', 'figure'),
    Output('boxplot-loan-amount', 'figure'),
    Output('distplot-loan-amount', 'figure'),
    Input('boxplot-applicant-income', 'clickData'),
    Input('distplot-applicant-income', 'clickData'),
    Input('boxplot-coapplicant-income', 'clickData'),
    Input('distplot-coapplicant-income', 'clickData'),
    Input('boxplot-loan-amount', 'clickData'),
    Input('distplot-loan-amount', 'clickData'),
)
def update_plots(clickData1, clickData2, clickData3, clickData4, clickData5, clickData6, clickData7):
    # Box Plot of Applicant Income
    fig_boxplot_applicant_income = px.box(df, x='ApplicantIncome', title='Box Plot of Applicant Income')
    
    # Distribution Plot of Applicant Income
    fig_distplot_applicant_income = px.histogram(df, x='ApplicantIncome', title='Distribution Plot of Applicant Income')
    
    # Box Plot of Coapplicant Income
    fig_boxplot_coapplicant_income = px.box(df, x='CoapplicantIncome', title='Box Plot of Coapplicant Income')
    
    # Distribution Plot of Coapplicant Income
    fig_distplot_coapplicant_income = px.histogram(df, x='CoapplicantIncome', title='Distribution Plot of Coapplicant Income')
    
    # Box Plot of Loan Amount
    fig_boxplot_loan_amount = px.box(df, x='LoanAmount', title='Box Plot of Loan Amount')
    
    # Distribution Plot of Loan Amount
    fig_distplot_loan_amount = px.histogram(df, x='LoanAmount', title='Distribution Plot of Loan Amount')

    return fig_boxplot_applicant_income, fig_distplot_applicant_income, fig_boxplot_coapplicant_income, fig_distplot_coapplicant_income, fig_boxplot_loan_amount, fig_distplot_loan_amount

if __name__ == '__main__':
    app.run_server(debug=True)
