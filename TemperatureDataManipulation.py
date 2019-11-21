import pandas as pd
from datetime import datetime

temp_data = pd.read_csv("temp.csv")  # read temp data
year = []  # declare lists
month = []
day = []
time = []
temp = []
for i in temp_data.itertuples():  # iterate through temperature data
    t = i[1].split(" ")  # get time
    date = t[0].split("-")  # get date
    year.append('20' + date[2])  # append relevant data
    day.append(date[0])
    month.append(datetime.strptime(date[1], "%b").month)
    time.append(t[1])
    temp.append(i[2])

temp_dict = {'Year': year, 'Month': month, 'Day': day, 'Time': time, 'Temp': temp}
df = pd.DataFrame(temp_dict)  # create DataFrame
df.to_csv("theatre_data.csv", index=False)  # create csv
