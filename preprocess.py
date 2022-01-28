import csv
import pandas as pd
from openpyxl import load_workbook

#################################  문장 구분 전처리 ################################
# file_path = "./data/wellness_dialog_for_autoregressive.txt"
# df=pd.read_csv('./data/ChatbotData.csv')
# # with open(file_path, 'r', encoding='utf-8') as f:
# #     a = csv.reader(f)
# #     df=pd.DataFrame(a)
# # print(f)
# print(df)
# df.drop(['label'], inplace=True, axis ='columns')
# print(df)
# df['QA'] = df['Q'] + '    '+ df['A']
# df.drop(['A'], inplace=True, axis ='columns')
# df.drop(['Q'], inplace=True, axis ='columns')
# print(df)
# print('a')
# df= pd.Series(df)
# print('b')
# final = []
# for i in df:
#     i = str(df.replace(',', ''))
#     final.append(i)
#
# df_final = pd.DataFrame(final)
  # 콤마제거 후 정수로 받음
# a = df.replace(',', '    ')

# df_final.to_csv('./data/wellness_dialog_for_autoregressive2.txt', index=False, header=None)
# file = open(file_path, 'r', encoding='utf-8')
# print(file)



###########비속어 전처리 ####################
import pandas as pd
with open("./data/비속어 데이터.txt", encoding='utf-8') as myfile:
    mydata = [line for line in myfile]
    df = pd.DataFrame(mydata, columns=['line'])

swear_word=[]
sep_ints=[]
for i in range(len(df)):

    word = df['line'][i][:-3]
    word = str(word.replace(',', ''))  # 콤마제거 후 정수로 받음
    word = str(word.replace('"', ''))
    sep_int= df['line'][i][-2:]
    print(sep_int)
    swear_word.append(word)
    sep_ints.append(sep_int)

print(sep_ints)
df_sep_ints = pd.Series(sep_ints)
print('df_sep_int:',df_sep_ints)
df_swear_word = pd.DataFrame(swear_word)
print(df_sep_ints)
print(df_swear_word.info())

swear_answer = '비속어를 쓰지 마세요.'
answer = '비속어가 아닙니다.'
print('type:',type(df_sep_ints))
answer_word = []
for d,i in enumerate(df_sep_ints):

    if i == '1\n':
        i = swear_answer
       # i = i+ ' ' + swear_answer
    elif i == '0\n':
        i = answer
        df_sep_ints.drop(index=d, axis=0, inplace=True)
        df_swear_word.drop(index=d, axis=0, inplace=True)
        # i = i +' ' + answer
    answer_word.append(i)

df_answer= pd.DataFrame(answer_word)
final_answer = df_swear_word[0]+"    "+df_answer[0]
print(type(final_answer))


final_answer.drop(final_answer[0]=="", inplace=True)
print(len(final_answer))
final_answer.to_csv('./data/wellness_dialog_for_autoregressive_swear_test.txt', index=False, header=None)
