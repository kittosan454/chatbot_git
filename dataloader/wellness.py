import torch
import torch.nn as nn
from torch.utils.data import Dataset # 데이터로더

from kogpt2_transformers import get_kogpt2_tokenizer
from kobert_transformers import get_tokenizer

class WellnessAutoRegressiveDataset(Dataset):
  """Wellness Auto Regressive Dataset"""

  def __init__(self,
               file_path = "../data/wellness_dialog_for_autoregressive2.txt",
               n_ctx = 1024
               ):
    self.file_path = file_path
    self.data =[]
    self.tokenizer = get_kogpt2_tokenizer()


    bos_token_id = [self.tokenizer.bos_token_id]
    # print(bos_token_id)
    eos_token_id = [self.tokenizer.eos_token_id]
    # print(eos_token_id)
    pad_token_id = [self.tokenizer.pad_token_id]

    file = open(self.file_path, 'r', encoding='utf-8')

    while True:
      line = file.readline()
      if not line:
        break
      datas = line.split("    ") # 엔터가 포함되어 있기 때문에 [1][:-1]이다
      # print(datas)
      index_of_words = bos_token_id +self.tokenizer.encode(datas[0]) + eos_token_id + bos_token_id + self.tokenizer.encode(datas[1][:-1])+ eos_token_id
      # print(index_of_words)

      pad_token_len = n_ctx - len(index_of_words)

      index_of_words += pad_token_id * pad_token_len

      self.data.append(index_of_words)

    file.close()

  def __len__(self):
    return len(self.data)

  def __getitem__(self,index):
    item = self.data[index]
    return item

class WellnessTextClassificationDataset(Dataset): # 텍스트 분류 모델
  """Wellness Text Classification Dataset"""
  def __init__(self,
               file_path = "../data/wellness_dialog_for_text_classification.txt",
               num_label = 359,
               device = 'cuda',
               max_seq_len = 512, # KoBERT max_length
               tokenizer = None
               ):
# 'cpu'
    self.file_path = file_path
    self.device = device
    self.data =[]
    self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()


    file = open(self.file_path, 'r', encoding='utf-8')
#################### 전처리 ##########
    while True:
      line = file.readline()
      if not line:
        break
      datas = line.split("    ")
      index_of_words = self.tokenizer.encode(datas[0]) # 토크나이저를 받아온다. 그리고 토크나이저 값으로 변경한다.
      token_type_ids = [0] * len(index_of_words) # 토크나이저의 길이만큼 0으로 채운 객체를 만듬, segment A
      attention_mask = [1] * len(index_of_words) # 토크나이저의 길이만큼 1로 채운 객체를 만든다

      # Padding Length
      padding_length = max_seq_len - len(index_of_words) # 패딩 할 길이 전체 받을 수 있는 길이에서 받은 길이를 뺀다.

      # Zero Padding
      index_of_words += [0] * padding_length # 남은 부분은 0으로 채운다.
      token_type_ids += [0] * padding_length
      attention_mask += [0] * padding_length

      # Label
      label = int(datas[1][:-1]) # 맨 오른쪽 값을 제외하고 모두(엔터 포함) 라벨 정수화
    ########### 데이터 텐서화 이후 딕셔너리로 바꾼다.
      data = {
              'input_ids': torch.tensor(index_of_words).to(self.device), # 입력층
              'token_type_ids': torch.tensor(token_type_ids).to(self.device), # 문장 segment층
              'attention_mask': torch.tensor(attention_mask).to(self.device), # 무엇이 패딩되었는지 알려준다. 실제 단어가 있으면 1임
              'labels': torch.tensor(label).to(self.device) # 라벨 값
             }
        # 딕셔너리 형태로 저장한다.
      self.data.append(data)
      print(self.data)
      # print("wellnes:",device)
    file.close()

  def __len__(self):
    return len(self.data)
  def __getitem__(self,index):
    item = self.data[index]
    return item

if __name__ == "__main__":
  dataset = WellnessAutoRegressiveDataset()
  dataset2 = WellnessTextClassificationDataset()
  print(dataset)
  print(dataset2)