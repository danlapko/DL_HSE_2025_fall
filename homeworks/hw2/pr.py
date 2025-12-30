import pandas as pd
data = pd.read_csv('train.csv')
print(len(data['ImageId'].unique()))