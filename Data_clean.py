import pandas as pd
df=pd.read_csv("IDS(BTP)/nsl-kdd/KDDTrain+.csv")
df.drop(columns=["difficulty"], inplace=True)
