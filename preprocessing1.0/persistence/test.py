import pandas as pd

url = './server/project03/p_data/train_sampled_V1.00.json'
df = pd.read_json(url,lines=True, orient='records')
print(df.head())
