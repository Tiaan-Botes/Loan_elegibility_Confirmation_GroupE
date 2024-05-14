import dash
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from DecisionTreeTrainer import classify_new_client

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
    html.H3("Enter client details"),

    html.Label("Dependents"),
    dcc.Input(id="dependents-input", type="number", value=0),

    html.Label("Applicant Income"),
    dcc.Input(id="applicant-income-input", type="number", value=0),

    html.Label("Coapplicant Income"),
    dcc.Input(id="coapplicant-income-input", type="number", value=0),
    
    html.Label("Loan Amount"),
    dcc.Input(id="loan-amount-input", type="number", value=0),
    html.Br(),
    html.Label("Loan Amount Term"),
    dcc.Input(id="loan-amount-term-input", type="number", value=0),

    html.Label("Credit History"),
    dcc.Input(id="credit-history-input", type="number", value=0),

    html.Label("Married"),
    dcc.Input(id="married-input", type="text", value=""),

    html.Label("Education"),
    dcc.Input(id="education-input", type="text", value=""),
    html.Br(),
    html.Label("Self Employed"),
    dcc.Input(id="self-employed-input", type="text", value=""),

    html.Label("\n Gender"),
    dcc.Input(id="gender-input", type="text", value=""),

    html.Label("Property Area"),
    dcc.Input(id="property-area-input", type="text", value=""),

    html.Button(id="submit-button", n_clicks=0, children="Submit")
    ]),

    html.Div(id="prediction-output"),

    html.Div([
        html.H3("Decision Tree Plot"),
        html.Img(src='https://raw.githubusercontent.com/Tiaan-Botes/Loan_elegibility_Confirmation_GroupE/master/decisiontree.png', style={'width':'100%'})
    ]),
    
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
        dcc.Graph(id='tree-plot')
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
    Output('tree-plot', 'figure'),
    Input('boxplot-applicant-income', 'clickData'),
    Input('distplot-applicant-income', 'clickData'),
    Input('boxplot-coapplicant-income', 'clickData'),
    Input('distplot-coapplicant-income', 'clickData'),
    Input('boxplot-loan-amount', 'clickData'),
    Input('distplot-loan-amount', 'clickData'),
    Input('tree-plot', 'clickData')
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
    
    # Train decision tree model
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    final_model_dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    final_model_dt.fit(X_train, y_train)


# Generate decision tree plot
    def plot_decision_tree(model):
    # Initialize figure
        fig = go.Figure()

    # Get decision tree attributes
        n_nodes = model.tree_.node_count
        children_left = model.tree_.children_left
        children_right = model.tree_.children_right
        feature = model.tree_.feature
        threshold = model.tree_.threshold

        # Add decision tree nodes
        for node_idx in range(n_nodes):
            if children_left[node_idx] != children_right[node_idx]:  # if not a leaf node
                feature_name = X.columns[feature[node_idx]]
                fig.add_trace(go.Scatter(x=[feature_name], y=[threshold[node_idx]],
                                     mode='markers+lines',
                                     marker=dict(size=20),
                                     line=dict(width=2, color='blue'),
                                     name=f'Node {node_idx}',
                                     text=f'Feature: {feature_name}<br>Threshold: {threshold[node_idx]:.2f}<br>Samples: {model.tree_.n_node_samples[node_idx]}',
                                     hoverinfo='text'
                                    ))

        # Update layout
        fig.update_layout(title='Decision Tree Plot',
                      xaxis=dict(title='Feature'),
                      yaxis=dict(title='Threshold'),
                      hovermode='closest')

        return fig


    # Decision Tree Plot
    fig_tree_plot = plot_decision_tree(final_model_dt)
    
    return fig_boxplot_applicant_income, fig_distplot_applicant_income, fig_boxplot_coapplicant_income, fig_distplot_coapplicant_income, fig_boxplot_loan_amount, fig_distplot_loan_amount, fig_tree_plot


############

@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('dependents-input', 'value'),
    State('applicant-income-input', 'value'),
    State('coapplicant-income-input', 'value'),
    State('loan-amount-input', 'value'),
    State('loan-amount-term-input', 'value'),
    State('credit-history-input', 'value'),
    State('married-input', 'value'),
    State('education-input', 'value'),
    State('self-employed-input', 'value'),
    State('gender-input', 'value'),
    State('property-area-input', 'value'),
)
def update_prediction(n_clicks, dependents, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, married, education, self_employed, gender, property_area):
    if n_clicks > 0:
        client_details_input = {
            'Dependents': dependents,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history,
            'Married': married,
            'Education': education,
            'Self_Employed': self_employed,
            'Gender': gender,
            'Property_Area': property_area
        }
        prediction = classify_new_client(client_details_input)
        if prediction == 0:
            return f"Predicted Loan Status: Sorry Your Application is likely to be denied :"
        else:
            return f"Predicted Loan Status: Congradulations Your Application is likely to be aproved!"



if __name__ == '__main__':
    app.run_server(debug=True)
