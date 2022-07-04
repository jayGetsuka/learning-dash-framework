from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from collections import OrderedDict

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

models = {'Regression': linear_model.LinearRegression,
        'Decision Tree': tree.DecisionTreeRegressor,
        'k-NN': neighbors.KNeighborsRegressor}

app.layout = html.Div([
    html.H1('Dash Tabs component demo'),
    dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
        dcc.Tab(label='Tab One', value='tab-1-example-graph'),
        dcc.Tab(label='Tab Two', value='tab-2-example-graph'),
        dcc.Tab(label='Tab There', value='tab-3-example-graph'),
    ]),
    html.Div(id='tabs-content-example-graph')
])

@app.callback(Output('tabs-content-example-graph', 'children'),
            Input('tabs-example-graph', 'value'))
def render_content(tab):
    df = pd.DataFrame(pd.read_excel("Outlook_homework.xlsx"))
    
    if tab == 'tab-1-example-graph':
        data = OrderedDict(pd.read_excel("Outlook_homework.xlsx"))
        df = pd.DataFrame(
            OrderedDict([(name, col_data) for (name, col_data) in data.items()])
            )
        return html.Div([
            html.H3('Tab content 1'),
            dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df.columns],
            )
        ])
    elif tab == 'tab-2-example-graph':
        fig = px.scatter(df, x="day", y="Humidity", color="Play")
        return html.Div([
            html.H3('Tab content 2'),
            dcc.Graph(
                id='graph-2-tabs-dcc',
                figure=fig
            )
        ])
    elif tab == 'tab-3-example-graph':
        X = df.Temperature.values[:, None]
        X_train, X_test, y_train, y_test = train_test_split(
            X, df.Humidity, random_state=42)

        model = models['Regression']()
        model.fit(X_train, y_train)

        x_range = np.linspace(X.min(), X.max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))
        
        return html.Div([
            html.H3('Tab content 3'),
            dcc.Graph(
                id='graph-3-tabs-dcc',
                figure = go.Figure([
                go.Scatter(x=X_train.squeeze(), y=y_train, 
                    name='train', mode='markers'),
                go.Scatter(x=X_test.squeeze(), y=y_test, 
                    name='test', mode='markers'),
                go.Scatter(x=x_range, y=y_range, 
                    name='prediction')
        ])
            )
        ])
        
        
        
if __name__ == '__main__':
    app.run_server(debug=True)