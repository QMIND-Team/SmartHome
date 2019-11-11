# use pandas instead?
import csv

file = open("en_climate_hourly_ON_6104152_11-2019_P1H.csv")
file.readline()
outFile = open("november_temperatures.csv", "w")
writer = csv.writer(outFile)
reader = csv.reader(file)

for line in reader:
    temperature = []
    time = []
    d = line[4].split()
    date = d[0]
    writer.writerow()  # continue

file.close()
outFile.close()
