import csv
import pandas as pd
from openpyxl import load_workbook

file_path = "./data/wellness_dialog_for_autoregressive.txt"
df=pd.read_csv('./data/ChatbotData.csv')
# with open(file_path, 'r', encoding='utf-8') as f:
#     a = csv.reader(f)
#     df=pd.DataFrame(a)
# print(f)
print(df)
df.drop(['label'], inplace=True, axis ='columns')
print(df)
df['QA'] = df['Q'] + '    '+ df['A']
df.drop(['A'], inplace=True, axis ='columns')
df.drop(['Q'], inplace=True, axis ='columns')
# a = df.replace(',', '    ')
print(df)
df.to_csv('./data/wellness_dialog_for_autoregressive2.txt', index=False, sep = '\t')
# file = open(file_path, 'r', encoding='utf-8')
# print(file)


