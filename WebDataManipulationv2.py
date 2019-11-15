import pandas as pd
import glob

for name in glob.glob("en_climate*"):
    with open("web_data.csv", "a") as f:
        data = pd.read_csv(name, usecols=['Year', 'Month', 'Day', 'Time', 'Temp (°C)', 'Wind Chill'])
        if glob.glob("en_climate*").index(name) == 0:
            data.to_csv(f, index=False)
        else:
            data.to_csv(f, header=False, index=False)
