import pandas as pd
import requests
from bs4 import BeautifulSoup
from csv import writer


page = requests.get('https://weather.gc.ca/forecast/hourly/on-69_metric_e.html')

soup = BeautifulSoup(page.text, 'html.parser')

time = soup.find_all('td', headers='header1')
temp = soup.find_all('td', headers='header2')

timeData = []
tempData = []

for i in range(len(temp)):
    timeData.append(time[i].text)
    tempData.append(temp[i].text)


weatherData = pd.DataFrame({
    'time': timeData,
    'temp': tempData,
})
    
weatherData.to_csv('ai_weather_data.csv')
