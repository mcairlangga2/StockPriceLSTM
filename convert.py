import pandas as pd

df = pd.read_csv('prediction_results.csv')
df.to_excel('prediction_results.xlsx')