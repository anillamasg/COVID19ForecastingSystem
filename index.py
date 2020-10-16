import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import requests
import os
import datetime
import time

from datetime import date
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# metricDictionary for various purposes
metricDictionary = {
    'confirmed': 'Confirmed Cases',
    'recovered': 'Recovered Cases',
    'deaths': 'Deaths',
    'NA': 'Australia',
    'ACT': 'Australian Capital Territory',
    'NSW': 'New South Wales',
    'NT': 'Northern Territory',
    'QLD': 'Queensland',
    'SA': 'South Australia',
    'TAS': 'Tasmania',
    'WA': 'Western Australia',
    'VIC': 'Victoria',

}

# locations in abbreviated form
locations = ['NA', 'ACT', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'WA', 'VIC']

# locations in abbreviated form besides Australia (NA)
compareLocations = ['ACT', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'WA', 'VIC']

# names of all metrics
allMetrics = ['confirmed', 'recovered', 'deaths']


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# prediction for each cases
def cases_prediction(location, metric):
    # Read data from datasource.
    df = pd.read_csv('COVID_AU_cumulative.csv')

    # check if location is Australia or else.
    if location == 'NA':
        dfForCases = df.loc[df['state_abbrev'].isnull()]
    else:
        dfForCases = df.loc[df['state_abbrev'] == location]

    # observed cases and corresponding dates
    cases = dfForCases[metric]
    dates = dfForCases['date']

    casesList = cases.values.tolist()
    dateList = dates.values.tolist()

    # call predict with location and metric
    predictedCases = predict(location, metric)

    # organize data for plotting.
    predictedDate = []
    lastDate = dateList[len(dateList) - 1]
    lastDateTime = datetime.strptime(lastDate, '%Y-%m-%d')

    for _ in range(10):
        lastDateTime += timedelta(days=1)
        predictedDate.append(lastDateTime.strftime("%Y-%m-%d"))

    # create and plot graph
    fig = make_subplots()
    fig.add_trace(
        go.Scatter(x=dateList, y=casesList, name="Total " + metricDictionary[metric],
                   marker=dict(color='#0400FF')))
    fig.add_trace(go.Scatter(x=predictedDate, y=predictedCases, name="Predicted " + metricDictionary[metric],
                             marker=dict(color='#EB752C')))

    fig.update_layout(hovermode="x",
                      title=metricDictionary[location],
                      title_font=dict(size=20),
                      xaxis_title='Date',
                      yaxis_title='Count',
                      font=dict(family='Overpass', size=12, color='#212121'),
                      autosize=True,
                      legend=dict(x=0, y=1, bordercolor='Black', borderwidth=1),
                      plot_bgcolor='#72f78c'
                      )

    # create html pages for location respective pages which is later rendered in an iframe
    if location == 'NA':
        fig.write_html(
            "plots/views/predicted/" + metricDictionary[metric].lower().replace(" ", "_") + "/" + metricDictionary[
                location] + "_prediction" + ".html")
    else:
        fig.write_html("plots/views/predicted/" + metricDictionary[metric].lower().replace(" ",
                                                                                           "_") + "/" + location + "_prediction" + ".html")


def compare_prediction(location1, location2, metric):
    # Read data from datasource.
    df = pd.read_csv('COVID_AU_cumulative.csv')
    df1 = df.loc[df['state_abbrev'] == location1]
    df2 = df.loc[df['state_abbrev'] == location2]

    # observed cases and corresponding dates
    cases1 = df1[metric]
    cases2 = df2[metric]
    dates = df1['date']

    casesList1 = cases1.values.tolist()
    casesList2 = cases2.values.tolist()
    dateList = dates.values.tolist()

    # organize data for plotting.
    predictedDate = []
    lastDate = dateList[len(dateList) - 1]
    lastDateTime = datetime.strptime(lastDate, '%Y-%m-%d')

    for _ in range(10):
        lastDateTime += timedelta(days=1)
        predictedDate.append(lastDateTime.strftime("%Y-%m-%d"))

    # call predict for each location with metric
    predictedCases1 = predict(location1, metric)
    predictedCases2 = predict(location2, metric)

    # create and plot graph
    fig = make_subplots()
    fig.add_trace(
        go.Scatter(x=dateList, y=casesList1, name="Total " + location1 + " " + metricDictionary[metric],
                   marker=dict(color='#0400FF')))
    fig.add_trace(
        go.Scatter(x=predictedDate, y=predictedCases1, name="Predicted " + location1 + " " + metricDictionary[metric],
                   marker=dict(color='#EB752C')))
    fig.add_trace(
        go.Scatter(x=dateList, y=casesList2, name="Total " + location2 + " " + metricDictionary[metric],
                   marker=dict(color='#F600FA')))
    fig.add_trace(
        go.Scatter(x=predictedDate, y=predictedCases2, name="Predicted " + location2 + " " + metricDictionary[metric],
                   marker=dict(color='#D9FA00')))

    fig.update_layout(hovermode="x",
                      title=metricDictionary[location1] + ' and ' + metricDictionary[location2],
                      title_font=dict(size=20),
                      xaxis_title='Date',
                      yaxis_title='Count',
                      font=dict(family='Overpass', size=12, color='#212121'),
                      autosize=True,
                      legend=dict(x=0, y=1, bordercolor='Black', borderwidth=1),
                      plot_bgcolor='#72f78c'
                      )

    # create html pages for location respective pages which is later rendered in an iframe
    if location1 < location2:
        fig.write_html("plots/views/compare/" + metricDictionary[metric].lower().replace(" ",
                                                                                         "_") + "/" + location1 + "_" + location2 + "_comparison" + ".html")
    else:
        fig.write_html("plots/views/compare/" + metricDictionary[metric].lower().replace(" ",
                                                                                         "_") + "/" + location2 + "_" + location1 + "_comparison" + ".html")


# this method is used to predict all cases from each metric.
# it takes location and metric parameters.
# ex: location=NSW, metric=confirmed
def predict(location, metric):
    # Read data from datasource.
    df = pd.read_csv('COVID_AU_cumulative.csv')

    # check if location is Australia or else.
    if location == 'NA':
        dfForCases = df.loc[df['state_abbrev'].isnull()]
    else:
        dfForCases = df.loc[df['state_abbrev'] == location]

    df1 = dfForCases[metric]

    # Apply MinMax scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

    # Splitting dataset into train and test split
    training_size = int(len(df1) * 0.80)
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

    # Reshape into X=a,a+1,a+2,a+3 and Y=a+4
    time_step = 50
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # LSTM model construction
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    print(model.summary())

    # Fitting model
    model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=200, batch_size=64, verbose=1)

    # Prediction
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Transforming into original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Calculate RMSE performance metrics
    # RMSE for Training Data
    math.sqrt(mean_squared_error(y_train, train_predict))

    # RMSE for Test Data
    math.sqrt(mean_squared_error(ytest, test_predict))

    # Shift train predictions for plotting
    look_back = 50
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict
    # Plot baseline and predictions
    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

    x_input = test_data[(len(test_data) - time_step):].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    # Demonstrate prediction for next 10 days

    lst_output = []
    n_steps = 50
    i = 0
    while (i < 10):

        if (len(temp_input) > 50):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i + 1

    output = scaler.inverse_transform(lst_output).tolist()
    real_output = []
    for output_list in output:
        real_output.append(int(output_list[0]))
    return real_output

# Downloads data from source
def download_data_file():
    url = 'https://raw.githubusercontent.com/M3IT/COVID-19_Data/master/Data/COVID_AU_cumulative.csv'
    r = requests.get(url, allow_redirects=True)
    open('download.csv', 'wb').write(r.content)
    os.remove("COVID_AU_cumulative.csv")
    os.rename(r'download.csv', r'COVID_AU_cumulative.csv')


if __name__ == "__main__":
    currentDataDate = datetime(2020, 10, 16).date()

    # never ending loop
    while True:
        tempCompareLocations = compareLocations
        today = date.today()

        # Checks if data retrieval date is earlier to today's date
        if today > currentDataDate:
            currentDataDate = today
            download_data_file()

            # prediction for all metrics in all locations
            for location in locations:
                for met in allMetrics:
                    cases_prediction(location, met)

            # comparison prediction of each metric with other metrics in all locations
            for compareLocation in compareLocations:
                tempCompareLocations.pop(0)

                for tempCompareLocation in tempCompareLocations:
                    for met in allMetrics:
                        compare_prediction(compareLocation, tempCompareLocation, met)

        time.sleep(28800)
        print("Checking date after each 8 hours for data update.")
