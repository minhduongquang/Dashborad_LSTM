# import streamlit as st
from datetime import date
import yfinance as yf
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import plotly.express as px
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
server = app.server

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
print('TODAY: ', TODAY)
fig = px.line()
df = yf.download('GOOG', start="2023-08-20", end=TODAY)
df.head()
data_tbl = df


#IMPORT LSTM MODEL

def getHistoryData(ticker, start, end, interval="1m", dateString="%Y-%m-%d"):
    from datetime import timedelta, datetime
    # Only 7 days worth of 1m granularity data are allowed to be fetched per request.
    # Maximum days for 1m period is within last 30 days.
    yfticker = yf.Ticker(ticker)
    start, end = datetime.strptime(start, dateString), datetime.strptime(end, dateString)
    df = pd.DataFrame()
    res = []
    while start < end:
        if start + timedelta(days=7) <= end:
            tmp = start + timedelta(days=7)
            res.append(yfticker.history(start=start.strftime(dateString), end=tmp.strftime(dateString), interval="1m"))
            start = tmp
        else:
            res.append(yfticker.history(start=start.strftime(dateString), end=end.strftime(dateString), interval="1m"))
            start = end
    df = df.append(res)
    df.index.rename("Date", inplace=True)
    df.index = df.index.strftime("%Y-%m-%d")
    df["Date"]=pd.to_datetime(df.index,format="%Y-%m-%d")
    return df

def getAllDateData(ticker):
    yfticker = yf.Ticker(ticker)
    df = yfticker.history(period="max", interval="1d")
    df.index.rename("Date", inplace=True)
    df.index = df.index.strftime("%Y-%m-%d")
    df["Date"]=pd.to_datetime(df.index,format="%Y-%m-%d")
    return df

def predict(ticker):
    df_nse = getAllDateData(ticker)
    data=df_nse.sort_index(ascending=True,axis=0)
    new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])
    for i in range(0,len(data)):
        new_data["Date"][i]=data['Date'][i]
        new_data["Close"][i]=data["Close"][i]
    new_data.index=new_data.Date
    new_data.drop("Date",axis=1,inplace=True)
    dataset=new_data.values
    train=dataset[0:(len(dataset) // 4 * 3),:]
    valid=dataset[(len(dataset) // 4 * 3):,:]
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)
    x_train,y_train=[],[]
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train,y_train=np.array(x_train),np.array(y_train)
    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    model=load_model("saved_model.h5")
    inputs=new_data[len(new_data)-len(valid)-60:].values
    inputs=inputs.reshape(-1,1)
    inputs=scaler.transform(inputs)
    X_test=[]
    # for i in range(60,inputs.shape[0]):
    #     X_test.append(inputs[i-60:i,0])
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test=np.array(X_test)
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    closing_price=model.predict(X_test)
    closing_price=scaler.inverse_transform(closing_price)
    train=new_data[:(len(new_data) // 4 * 3)]
    valid=new_data[(len(new_data) // 4 * 3):]
    valid['Predictions']=closing_price
    return train, valid

#END
app.layout = html.Div([
    # Display the navbar
        html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
        dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Predict Stock by LSTM',children=[
            html.Div([
                # dcc.Input(
                #     id="input_string",
                #     type="text",
                #     placeholder="Ticker name",
                #     value="AAPL",
                #     style={"textAlign": "center"}
                # ),
                dcc.Dropdown(
                    id='input_string',
                    options=[
                        {'label': 'GOOG', 'value': 'GOOG'},
                        {'label': 'AAPL', 'value': 'AAPL'},
                        {'label': 'MSFT', 'value': 'MSFT'},
                        {'label': 'AMZN', 'value': 'AMZN'},
                        {'label': 'TSLA', 'value': 'TSLA'},
                        {'label': 'META', 'value': 'META'},
                        {'label': 'NVDA', 'value': 'NVDA'},
                        {'label': 'NFLX', 'value': 'NFLX'},
                        {'label': 'INTC', 'value': 'INTC'},
                        {'label': 'ADBE', 'value': 'ADBE'},
                        {'label': 'PEP', 'value': 'PEP'},
                        {'label': 'SBUX', 'value': 'SBUX'}
                    ],
                    value='GOOG',
                    searchable=True,
                    placeholder="Search stocks..."
                ),
                html.H2("Actual closing price",style={"textAlign": "center"}),
                dcc.Graph(id="ActualData"),
                html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
                dcc.Graph(id="PredictedData")                
            ])                
        ]),
        dcc.Tab(label='Visualize Stock Data', children=[
            html.Div([
                html.Label('Select dataset for visualization: '),
                dcc.Dropdown(
                    id='input-ticker',
                    options=[
                        {'label': 'GOOG', 'value': 'GOOG'},
                        {'label': 'AAPL', 'value': 'AAPL'},
                        {'label': 'MSFT', 'value': 'MSFT'},
                        {'label': 'AMZN', 'value': 'AMZN'},
                        {'label': 'TSLA', 'value': 'TSLA'},
                        {'label': 'META', 'value': 'META'},
                        {'label': 'NVDA', 'value': 'NVDA'},
                        {'label': 'NFLX', 'value': 'NFLX'},
                        {'label': 'INTC', 'value': 'INTC'},
                        {'label': 'ADBE', 'value': 'ADBE'},
                        {'label': 'PEP', 'value': 'PEP'},
                        {'label': 'SBUX', 'value': 'SBUX'}
                    ],
                    value='GOOG',
                    searchable=True,
                    placeholder="Search stocks..."
                ),
                html.Label('Years of prediction:'),
                html.Div(id='output-data-load'),
                html.H2("Actual Data"),
                dcc.Graph(id='raw-data-plot'),
                html.Label('Movement Price: '),
                dash_table.DataTable(data_tbl.to_dict('records'), [{"name": i, "id": i} for i in data_tbl.columns])
            ])
        ])
    ])
])

@app.callback(
    [Output('output-data-load', 'children'),
     Output('raw-data-plot', 'figure')],
    [Input('input-ticker', 'value')]
)
def update_output(selected_stock):
    data_load_state = 'Loading data...'
    data = load_data(selected_stock)
    data_load_state = 'Loading data... done!'

    fig = {
        'data': [
            {'x': data['Date'], 'y': data['Open'], 'type': 'scatter', 'name': 'stock_open'},
            {'x': data['Date'], 'y': data['Close'], 'type': 'scatter', 'name': 'stock_close'}
        ],
        'layout': {
            'title': 'Time Series data with Rangeslider',
            'xaxis': {'rangeslider': {'visible': True}}
        }
    }

    return data_load_state, fig

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    data.head()
    return data


@app.callback([Output('ActualData', 'figure'), 
              Output('PredictedData', 'figure')],
              [Input('input_string', 'value')])
def update_ticker_input(string_value):
    print("String Value: ", string_value)
    train, valid = predict(string_value)
    print(valid)
    figure1={
        "data":[
            go.Scatter(
                x=train.index,
                y=valid["Close"],
                mode='lines',
            )
        ],
        "layout":go.Layout(
            xaxis={'title':'Date'},
            yaxis={'title':'Closing Rate'}
        )
    }
    # figure1 = {
    #     'data': [
    #         # {'x': train['Date'], 'y': train['Open'], 'type': 'scatter', 'name': 'stock_open'},
    #         {'x': train['Date'], 'y': train['Close'], 'type': 'scatter', 'name': 'stock_close'}
    #     ],
    #     'layout': {
    #         'title': 'Time Series data with Rangeslider',
    #         'xaxis': {'rangeslider': {'visible': True}}
    #     }
    # }
    # figure2 = {
    #     'data': [
    #         # {'x': valid['Date'], 'y': valid['Open'], 'type': 'scatter', 'name': 'stock_open'},
    #         {'x': valid['Date'], 'y': valid['Close'], 'type': 'scatter', 'name': 'stock_close'}
    #     ],
    #     'layout': {
    #         'title': 'Time Series data with Rangeslider',
    #         'xaxis': {'rangeslider': {'visible': True}}
    #     }
    # }
    figure2={
        "data":[
            go.Scatter(
                x=valid.index,
                y=valid["Predictions"],
                mode='lines'
            )
        ],
        "layout":go.Layout(
            xaxis={'title':'Date'},
            yaxis={'title':'Closing Rate'}
        )
    }
    return [figure1, figure2]

if __name__ == '__main__':
    app.run_server(debug=True)