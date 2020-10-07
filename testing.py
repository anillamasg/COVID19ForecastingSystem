from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv('COVID_AU_state_cumulative.csv')
df1 = df.loc[df['state_abbrev'] == 'NSW']
confirmedCasesDf = df1['confirmed']
deathsDf = df1['deaths']
datesDf = df1['date']

dateList = datesDf.values.tolist()

print(datesDf)
print(dateList)
print(len(dateList))
print("Date type = ", type(dateList[0]))
print("Dates type = ", type(dateList))

predictedDate = ['2020-09-24', '2020-09-25', '2020-09-26', '2020-09-27', '2020-09-28', '2020-09-29', '2020-09-30']
predictedConfirmedCases = [4206, 4204, 4200, 4198, 4196, 4190, 4185]
confirmedCasesDfList = confirmedCasesDf.values.tolist()
print(confirmedCasesDfList)
print(type(confirmedCasesDfList[0]))
print(type(confirmedCasesDfList))


fig = make_subplots()
fig.add_trace(go.Scatter(x=dateList, y=confirmedCasesDfList, name="Confirmed (new cases)", marker=dict(color='#0400FF')))
fig.add_trace(go.Scatter(x=predictedDate, y=predictedConfirmedCases, name="Predicted Confirmed (new cases)", marker=dict(color='#EB752C')))

fig.update_layout(hovermode="x",
                  title='Australia',
                  title_font=dict(size=20),
                  xaxis_title='Date',
                  yaxis_title='Count',
                  font=dict(family='Overpass', size=12, color='#212121'),
                  autosize=True,
                  # width=1200,
                  # height=500,
                  legend=dict(x=0, y=0.9, bordercolor='Black', borderwidth=1),
                  plot_bgcolor='#72f78c',
                  )

# fig.add_trace(go.Scatter(x=dateList, y=deathsDf, name="Confirmed (new cases)", marker=dict(color='#EB752C')))
# fig.add_trace(go.Bar(x=dateList, y=deathsDf, name="Death (new cases)", marker=dict(color='#EB752C'), opacity=0.7,
#                      width=len(dateList), hoverlabel=dict(bgcolor='#EB752C')))

# fig.add_trace(go.Scatter(x=dates, y=newpredsave, name="Prediction (new cases)", marker=dict(color='#EB752C')))
# fig.add_trace(go.Bar(x=dates, y=newsave, name="True data (new cases)", marker=dict(color='#EB752C'), opacity=0.7, width=[1]*len(dates), hoverlabel=dict(bgcolor='#FF351C')))
# fig.add_trace(go.Scatter(x=dates, y=deadpredsave, name="Prediction (deaths)", marker=dict(color='#2D58BE')), secondary_y=True)
# fig.add_trace(go.Bar(x=dates, y=deadsave, name="True data (deaths)", marker=dict(color='#2D58BE'), opacity=0.3, width=[1]*len(dates), hoverlabel=dict(bgcolor='#1A22AB')), secondary_y=True)
# fig.update_layout(hovermode="x",
#                   title='Australia',
#                   title_x=0.5,
#                   title_font=dict(size=20),
#                   xaxis_title='Date',
#                   font=dict(family='Overpass', size=12, color='#212121'),
#                   yaxis_tickformat=',.0f',
#                   xaxis=dict(dtick=30),
#                   autosize=False,
#                   width=1200,
#                   height=500,
#                   legend=dict(x=0.6, y=0.9, bordercolor='Black', borderwidth=1),
#                   plot_bgcolor='rgba(0,0,0,0)',
#                   )

fig.write_html("plots/views/predicted/confirmed_cases_Australia_prediction" + ".html")
