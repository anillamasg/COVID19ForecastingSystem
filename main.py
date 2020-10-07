import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('COVID_AU_cumulative.csv')
# print(df.reset_index()['administrative_area_level_2'])
# print(np.nan)
# df1 = df.loc[pd.isna(df['administrative_area_level_2'])]
df1 = df.loc[df['state_abbrev'] == 'NSW']
confirmedCasesDf = df1['confirmed']
datesDf = df1['date']

left = df1['date'].iloc[0]
right = df1['date'].iloc[len(df1['date'])-1]


plt.plot(datesDf, confirmedCasesDf)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.gca().set_xbound(left, right)

plt.show()

print("asd "+df1['date'].iloc[0])
print("asd "+df1['date'].iloc[len(df1['date'])-1])
print(type(datesDf))


left = df1['date'].iloc[0]
right = df1['date'].iloc[len(df1['date'])-1]

print(type(left))
print(left)
print(right)

# Create scatter plot of Positive Cases
#
plt.scatter(
  datesDf, confirmedCasesDf, c="blue", edgecolor="black",
  linewidths=0.1, marker = ".", alpha = 0.8, label="Total Positive Tested"
)

# Format the date into months & days
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#
# # Change the tick interval
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
#
# # Puts x-axis labels on an angle
plt.gca().xaxis.set_tick_params(rotation = 30)
#
# # Changes x-axis range
plt.gca().set_xbound(left, right)

plt.show()
#
# plt.plot(datesDf, confirmedCasesDf)
# plt.show()