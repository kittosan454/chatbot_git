import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from tqdm import tqdm

import torch
from transformers import AdamW
from torch.utils.data import dataloader
from dataloader.wellness import WellnessTextClassificationDataset
from model.kobert import KoBERTforSequenceClassfication

def train(device, epoch, model, optimizer, train_loader, save_step, save_ckpt_path, train_step = 0):
    losses = []
    train_start_index = train_step+1 if train_step != 0 else 0
    total_train_step = len(train_loader) # 현재 에폭에 학습할 총 train step 데이터 갯수임
    model.train() # 신경망을 학습모드로 전환

# tqdm 진행바 : desc 진행바 앞에 텍스트 출력, total 전체 반복량 ,  실행과 종료를 위함
    with tqdm(total= total_train_step, desc=f"Train({epoch})") as pbar:
        pbar.update(train_step) # train_step update
        # train_start_index는 인덱스 시작위치임
        for i, data in enumerate(train_loader, train_start_index):

            optimizer.zero_grad() #gradient를 0으로 초기화
            outputs = model(**data)
            # print('outputs:',outputs)
            loss = outputs[0]

            losses.append(loss.item())

            loss.backward() # 역전파
            optimizer.step() # 가중치와 편향 업데이트

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(losses):.3f})")

            if i >= total_train_step or i % save_step == 0: # 모델 저장
                torch.save({
                    'epoch': epoch,  # 현재 학습 epoch
                    'model_state_dict': model.state_dict(),  # 모델 저장
                    'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
                    'loss': loss.item(),  # Loss 저장
                    'train_step': i,  # 현재 진행한 학습
                    'total_train_step': len(train_loader)  # 현재 epoch에 학습 할 총 train step
                }, save_ckpt_path)

    return np.mean(losses)



if __name__ == '__main__':
    data_path = "./data/wellness_dialog_for_text_classification_train.txt"
    checkpoint_path ="./checkpoint"
    save_ckpt_path = f"{checkpoint_path}/kobert-wellnesee-text-classification.pth"

    n_epoch = 20          # Num of Epoch
    batch_size = 4      # 배치 사이즈
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_step = 100 # 학습 저장 주기
    learning_rate = 5e-5  # Learning Rate

    # WellnessTextClassificationDataset 데이터 로더
    dataset = WellnessTextClassificationDataset()
    # 데이터 셋 인자와 배치사이즈 인자를 받는다. shuffle=True를 주면 epoch마다 데이터셋을 섞어서 학습된다.
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # koBERT 모델 불러오기
    model = KoBERTforSequenceClassfication()
    model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    # named_parameters 튜플로 wieght값과 bias값에 이름을 준다.
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # n_decay에서 nd라는 값을 뽑아서 만약 n안에 nd가 하나라도 있다면 True를 반환한다. 그리고 만약 이러한 반환이 False면 named_parameters()에서 파라미터값을 뽑아낸다.
    # 즉 'params': p 가 된다. 딕셔너리 형태인듯.리스트[딕셔너리{}]로 들어간다.
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    # 리스트나 딕셔너리 형태로 구분된 파라미터 그룹들이 들어간다.
    # 초기화
    pre_epoch, pre_loss, train_step = 0, 0, 0
    # check 포인트를 가져와 학습시킨다.
    if os.path.isfile(save_ckpt_path):
        checkpoint = torch.load(save_ckpt_path, map_location=device)
        pre_epoch = checkpoint['epoch'] # 이전 에폭 값
        pre_loss = checkpoint['loss'] # 이전 손실 값
        train_step =  checkpoint['train_step']
        total_train_step =  checkpoint['total_train_step']

        model.load_state_dict(checkpoint['model_state_dict']) # kobert 모델의 가중치와 편향치를 로드한다.
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # 최적화 함수의 가중치와 편향치를 로드한다.

        print(f"load pretrain from: {save_ckpt_path}, epoch={pre_epoch}, loss={pre_loss}")
        # best_epoch += 1

    losses = []
    offset = pre_epoch
    for step in range(n_epoch):
        epoch = step + offset
        loss = train(device, epoch, model, optimizer, train_loader, save_step, save_ckpt_path, train_step)
        losses.append(loss)

    # data
    data = {
        "loss": losses
    }
    df = pd.DataFrame(data)
    display(df)

    # graph
    plt.figure(figsize=[12, 4])
    plt.plot(losses, label="loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


