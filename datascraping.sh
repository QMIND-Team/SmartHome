#!/bin/bash

for year in `seq 2009 2019`; do 
  for month in `seq 1 12`; do 
    wget --content-disposition "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=47267&Year=${year}&Month=${month}&Day=14&timeframe=1&submit= Download+Data" 
    echo $year
    echo $month
    echo ""

  done
done

mkdir data

for f in *.csv; do
  mv $f  data/$f
done

zip -r data.zip data

for f in data/*.csv; do
  rm $f
done

rmdir data
