from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

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

def cases_prediction(location, metric):
    df = pd.read_csv('COVID_AU_cumulative.csv')
    if location == 'NA':
        dfForCases = df.loc[df['state_abbrev'].isnull()]
    else:
        dfForCases = df.loc[df['state_abbrev'] == location]

    cases = dfForCases[metric]
    dates = dfForCases['date']

    casesList = cases.values.tolist()
    dateList = dates.values.tolist()

    predictedCases = [26942, 26942, 26942, 26942, 26942, 26942, 26942, 26942, 26942, 26942]  # confirmed_AU_cases
    # predictedCases = [24026, 24030, 24031, 24031, 24036, 24046, 24055, 24056, 24062, 24068] #recovered_AU_cases
    # predictedCases = [816, 816, 818, 821, 823, 823, 826, 826, 826, 826] #deaths_AU_cases

    # predictedCases = [113, 115, 115, 115, 117, 117, 118, 120, 121, 123] #confirmed_ACT_cases
    # predictedCases = [110, 110, 114, 114, 114, 114, 114, 114, 114, 114] #recovered_ACT_cases
    # predictedCases = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3] #deaths_ACT_cases

    # predictedCases = [4206, 4207, 4209, 4211, 4212, 4212, 4215, 4219, 4229, 4230] #confirmed_NSW_cases
    # predictedCases = [2802, 2805, 2805, 2810, 2812, 2812, 2812, 2814, 2819, 2820] #recovered_NSW_cases
    # predictedCases = [52, 52, 52, 52, 52, 52, 52, 52, 52, 53] #deaths_NSW_cases

    # predictedCases = [33, 33, 33, 33, 33, 33, 33, 33, 33, 33] #confirmed_NT_cases
    # predictedCases = [33, 33, 33, 33, 33, 33, 33, 33, 33, 33] #recovered_NT_cases
    # predictedCases = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #deaths_NT_cases

    # predictedCases = [1153, 1153, 1155, 1155, 1156, 1158, 1158, 1158, 1158, 1158] #confirmed_QLD_cases
    # predictedCases = [1124, 1124, 1124, 1125, 1125, 1125, 1126, 1126, 1127, 1127] #recovered_QLD_cases
    # predictedCases = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6] #deaths_QLD_cases

    # predictedCases = [466, 466, 466, 466, 466, 466, 466, 466, 466, 466] #confirmed_SA_cases
    # predictedCases = [462, 462, 463, 463, 463, 464, 464, 464, 464, 464] #recovered_SA_cases
    # predictedCases = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4] #deaths_SA_cases

    # predictedCases = [230, 230, 230, 230, 230, 230, 230, 230, 230, 230] #confirmed_TAS_cases
    # predictedCases = [217, 217, 217, 217, 218, 218, 218, 218, 218, 220] #recovered_TAS_cases
    # predictedCases = [13, 13, 13, 13, 13, 13, 13, 13, 13, 13] #deaths_TAS_cases

    # predictedCases = [20076, 20090, 20106, 20120, 20146, 20160, 20176, 20200, 20216, 20236] #confirmed_VIC_cases
    # predictedCases = [18628, 18670, 18706, 18734, 18768, 18828, 18855, 18886, 18920, 18979] #recovered_VIC_cases
    # predictedCases = [729, 729, 729, 729, 732, 732, 732, 732, 732, 733] #deaths_VIC_cases

    # predictedCases = [665, 666, 668, 668, 668, 668, 670, 670, 670, 670] #confirmed_WA_cases
    # predictedCases = [650, 655, 655, 655, 655, 655, 660, 660, 660, 660] #recovered_WA_cases
    # predictedCases = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9] #deaths_WA_cases

    predictedDate = []
    lastDate = dateList[len(dateList) - 1]
    lastDateTime = datetime.strptime(lastDate, '%Y-%m-%d')

    for _ in range(10):
        lastDateTime += timedelta(days=1)
        predictedDate.append(lastDateTime.strftime("%Y-%m-%d"))

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

    if location == 'NA':
        fig.write_html(
            "plots/views/predicted/" + metricDictionary[metric].lower().replace(" ", "_") + "/" + metricDictionary[
                location] + "_prediction" + ".html")
    else:
        fig.write_html("plots/views/predicted/" + metricDictionary[metric].lower().replace(" ",
                                                                                           "_") + "/" + location + "_prediction" + ".html")


# cases_prediction('QLD', 'confirmed')
# cases_prediction('WA', 'recovered')
# cases_prediction('TAS', 'deaths')

def compare_prediction(location1, location2, metric):
    df = pd.read_csv('COVID_AU_cumulative.csv')
    df1 = df.loc[df['state_abbrev'] == location1]
    df2 = df.loc[df['state_abbrev'] == location2]

    cases1 = df1[metric]
    cases2 = df2[metric]
    dates = df1['date']

    casesList1 = cases1.values.tolist()
    casesList2 = cases2.values.tolist()
    dateList = dates.values.tolist()

    predictedDate = []
    lastDate = dateList[len(dateList) - 1]
    lastDateTime = datetime.strptime(lastDate, '%Y-%m-%d')

    for _ in range(10):
        lastDateTime += timedelta(days=1)
        predictedDate.append(lastDateTime.strftime("%Y-%m-%d"))

    # predictedCases1 = [113, 115, 115, 115, 117, 117, 118, 120, 121, 123] #confirmed_ACT_cases
    # predictedCases1 = [110, 110, 114, 114, 114, 114, 114, 114, 114, 114] #recovered_ACT_cases
    # predictedCases1 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3] #deaths_ACT_cases

    # predictedCases1 = [4206, 4207, 4209, 4211, 4212, 4212, 4215, 4219, 4229, 4230] #confirmed_NSW_cases
    # predictedCases1 = [2802, 2805, 2805, 2810, 2812, 2812, 2812, 2814, 2819, 2820] #recovered_NSW_cases
    # predictedCases1 = [52, 52, 52, 52, 52, 52, 52, 52, 52, 53] #deaths_NSW_cases

    # predictedCases1 = [33, 33, 33, 33, 33, 33, 33, 33, 33, 33] #confirmed_NT_cases
    # predictedCases1 = [33, 33, 33, 33, 33, 33, 33, 33, 33, 33] #recovered_NT_cases
    # predictedCases1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #deaths_NT_cases

    # predictedCases1 = [1153, 1153, 1155, 1155, 1156, 1158, 1158, 1158, 1158, 1158] #confirmed_QLD_cases
    # predictedCases1 = [1124, 1124, 1124, 1125, 1125, 1125, 1126, 1126, 1127, 1127] #recovered_QLD_cases
    # predictedCases1 = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6] #deaths_QLD_cases

    # predictedCases1 = [466, 466, 466, 466, 466, 466, 466, 466, 466, 466] #confirmed_SA_cases
    # predictedCases1 = [462, 462, 463, 463, 463, 464, 464, 464, 464, 464] #recovered_SA_cases
    # predictedCases1 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4] #deaths_SA_cases

    # predictedCases1 = [230, 230, 230, 230, 230, 230, 230, 230, 230, 230] #confirmed_TAS_cases
    # predictedCases1 = [217, 217, 217, 217, 218, 218, 218, 218, 218, 220] #recovered_TAS_cases
    # predictedCases1 = [13, 13, 13, 13, 13, 13, 13, 13, 13, 13] #deaths_TAS_cases

    # predictedCases1 = [20076, 20090, 20106, 20120, 20146, 20160, 20176, 20200, 20216, 20236] #confirmed_VIC_cases
    # predictedCases1 = [18628, 18670, 18706, 18734, 18768, 18828, 18855, 18886, 18920, 18979] #recovered_VIC_cases
    predictedCases1 = [729, 729, 729, 729, 732, 732, 732, 732, 732, 733] #deaths_VIC_cases

    # predictedCases2 = [665, 666, 668, 668, 668, 668, 670, 670, 670, 670] #confirmed_WA_cases
    # predictedCases2 = [650, 655, 655, 655, 655, 655, 660, 660, 660, 660] #recovered_WA_cases
    predictedCases2 = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9] #deaths_WA_cases

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

    if location1 < location2:
        fig.write_html("plots/views/compare/" + metricDictionary[metric].lower().replace(" ",
                                                                                         "_") + "/" + location1 + "_" + location2 + "_comparison" + ".html")
    else:
        fig.write_html("plots/views/compare/" + metricDictionary[metric].lower().replace(" ",
                                                                                         "_") + "/" + location2 + "_" + location1 + "_comparison" + ".html")


