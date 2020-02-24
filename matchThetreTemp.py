import os
CWD = os.getcwd()

thetreData = open('theatre_data.csv', 'r')
weatherData = open('2019webdata.csv', 'r')



thetreData.close()
weatherData.close()

outputFile = open('trainingData.txt','w+')
outputFile.close()