def readTheatreData(fileName):  
    with open(fileName) as theatreData:
        firstLine = True
        myDict = {}
        for line in theatreData:
            if(firstLine):
                firstLine = False
                continue
            if(line == '\n'):
                continue
            line = line.split(',')
            y = line[0]
            m = line[1]
            d = line[2]
            t = line[3]
            temp = float(line[4].rstrip('\n'))
            myDict[y,m,d,t] = temp
    return myDict

def readWeatherData(fileName):
    with open(fileName) as weatherData:
        firstLine = True
        myDict = {}
        index = 0
        for line in weatherData:
            if(firstLine):
                firstLine = False
                index += 1
                continue
            if(line == '\n'):
                index += 1
                continue
            line = line.split(',')
            y = line[0]
            m = line[1]
            d = line[2]
            t = line[3]
            myDict[y,m,d,t] = index
            index += 1
    return myDict

weatherDict = readWeatherData('2019webdata.csv')
theatreDict = readTheatreData('theatre_data.csv')

with open('2019webdata.csv') as weatherFile:
    weatherData = weatherFile.readlines()
    for i in range(1, len(weatherData)):
        weatherData[i] = weatherData[i].split(',')
        weatherData[i] = float(weatherData[i][4].rstrip('\n'))
outputFile = open('trainingData.txt', 'w+')

for key in theatreDict:
    if(key in weatherDict):
        index = weatherDict[key]
        for i in range(24):
            outputFile.write(str(weatherData[index + i]) + ',')
        outputFile.write(str(theatreDict[key]) + '\n')

outputFile.close()


