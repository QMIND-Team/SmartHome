import csv
import os 

CWD= os.getcwd()

def get_web_data():
    
    with open(CWD +'\\web_dataV2.csv', 'r', newline='') as f:
        csv_reader = csv.reader(f)
        web_data_list = list(csv_reader)
    return web_data_list

def get_temp_and_time(web_data_list):
    temp_list = []
    time_list = []
    for i in range(1, len(web_data_list), 2):
        temp = web_data_list[i][4]
        if temp == '':
            temp = '0.0'
            
        temp = list(temp)
            
        if temp[0] == "-":
            temp[0] = ''
            temp = ''.join(temp)
            temp = -1*float(temp)
        else:
            temp = ''.join(temp)
            temp = float(temp)
        temp_list.append(temp)
        
        time = web_data_list[i][3]
        if time[1] == ":":
            time = ("0" + str(time))
        time = list(time)
        time[3] = '3'
        time = ''.join(time)
        time_list.append(time)
            

    return [temp_list, time_list]
        
def get_average(tempList):
    tempList2 = tempList
    tempList2.append(0)
    temp_average = []
    temp1 = 0
    temp2 = 0

    for i in range(len(tempList)):
        temp1 = tempList[i]
        temp2 = tempList2[i]
        temp_average.append((temp1 + temp2)/2)

    return temp_average

def new_line(tempTime, tempAve, webData):
    time_list = tempTime[1]
    new_line_list = []
    previous_line = []

    for i in range(1, len(webData), 2):
        for j in range(len(time_list)):
            previous_line = webData[i]
            previous_line[3] = time_list[j]
            previous_line[4] = tempAve[j]
            new_line_list.append(previous_line)
    return new_line_list

def new_web_data(newLineList, webData):

    for i in range(2, len(webData), 2):
        for j in range(len(newLineList)):
            webData[i] = newLineList[j]
    return webData

def updated_web_data(newWebData):
    new_csv = 'web_dataV3.csv'
    with open(new_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(newWebData)):
            writer.writerow(newWebData[i])

def main():
    webData = get_web_data()
    tempTime = get_temp_and_time(webData)
    temp = get_average(tempTime[0])
    time = tempTime[1]
    newLineList = new_line(tempTime, temp, webData)
    newWebData = new_web_data(newLineList, webData)
    updatedWebData = updated_web_data(newWebData)

main()

        


    
        
