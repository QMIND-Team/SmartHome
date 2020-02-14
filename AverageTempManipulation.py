import os

CWD = os.getcwd()

dataFile = open( 'web_data.csv', 'r')
dataOutput = open('webDataV2.csv','w+')
header = True
fLine = True

for line in dataFile:
    if (header):
        header = False
        continue

    if (fLine): 
        prevLine = line.split(',')
        year = prevLine[0]
        month = prevLine[1]
        day = prevLine[2]
        time = prevLine[3]
        temp = prevLine[4].rstrip("\n")
        continue

    line = line.split(',')

    nYear = line[0]
    nMonth = line[1]
    nDay = line[2]
    nTime = line[3]
    nTemp = line[4].rstrip("\n")

    nTime = nTime[:-2] + '30'
    nTemp = str((int(temp) + int(nTemp)) / 2)

    outputA = (year + ',' + month + ',' + day + ',' + time + ',' + temp + '\n')

    outputB = (nYear + ',' + nMonth + ',' + nDay + ',' + nTime + ',' + nTemp + '\n')

    year = nYear
    month = nMonth
    day = nDay
    time = nTime
    temp = nTemp

    dataFile.write(outputA)
    dataFile.write(outputB)

dataOutput.close()
dataFile.close()
