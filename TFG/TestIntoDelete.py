import pandas as pd

data1 = [0,1,2,3,4]
data2 = [1,2,3,4,5]
df = pd.DataFrame([data2,data1])
df = df.transpose()
df.columns = ['real','pred']
print(df)