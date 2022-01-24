import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import dataloader
from dataloader.wellness import WellnessAutoRegressiveDataset
from model.kogpt2 import DialogKoGPT2


def train(device, epoch, model, optimizer, train_loader, save_step, save_ckpt_path, count = 0):
    losses = []
    train_start_index = count+1 if count != 0 else 0
    total_train_step = len(train_loader) # 현재 에폭에 학습할 총 train step 데이터 갯수임
    model.train() # 신경망을 학습모드로 전환

# tqdm 진행바 : desc 진행바 앞에 텍스트 출력, total 전체 반복량 ,  실행과 종료를 위함
    with tqdm(total= total_train_step, desc=f"Train({epoch})") as pbar:
        pbar.update(train_step) # train_step update
        # train_start_index는 인덱스 시작위치임
        for i, data in enumerate(train_loader, train_start_index):

            optimizer.zero_grad() #gradient를 0으로 초기화
            # print('outputs:',outputs)

            ########

            data = torch.stack(data)  # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다. 텐서를 쌓아올린다.
            data = data.transpose(1, 0) # 순서를 바꿔준다.
            data = data.to(ctx)

            outputs = model(data, labels=data)
            _, logits = outputs[:2]

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = data[..., 1:].contiguous()

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # if count % 10 == 0:
            #     print('epoch no.{} train no.{}  loss = {}'.format(epoch, count + 1, loss))
            # if (count > 0 and count % save_step == 0) or (len(data) < batch_size):
            #     torch.save({
            #         'epoch': epoch,
            #         'train_no': count,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': loss
            #     }, save_ckpt_path)
            # count += 1


            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(losses):.3f})")

            if i >= total_train_step or i % save_step == 0: # 모델 저장
                torch.save({
                    'epoch': epoch,  # 현재 학습 epoch
                    'model_state_dict': model.state_dict(),  # 모델 저장
                    'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
                    'loss': loss,  # Loss 저장
                    'train_no': count,  # 현재 진행한 학습
                    'total_train_step': len(train_loader)  # 현재 epoch에 학습 할 총 train step
                }, save_ckpt_path)
            count += 1
    return np.mean(losses)



if __name__ == '__main__':
    data_path = "./data/wellness_dialog_for_autoregressive2.txt"
    checkpoint_path ="../checkpoint"
    save_ckpt_path = f"{checkpoint_path}/kogpt2-wellnesee-auto-regressive.pth"

    n_epoch = 5         # Num of Epoch
    batch_size = 1      # 배치 사이즈
    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ctx)

    save_step = 100 # 학습 저장 주기
    learning_rate = 5e-5  # Learning Rate

    dataset= WellnessAutoRegressiveDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DialogKoGPT2()
    model.to(device)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=3) # 크로스 엔트로피 레이어 층
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    pre_epoch, pre_loss, train_step = 0, 0, 0
    # check 포인트를 가져와 학습시킨다.
    if os.path.isfile(save_ckpt_path):
        print(save_ckpt_path)
        checkpoint = torch.load(save_ckpt_path, map_location=device)
        pre_epoch = checkpoint['epoch'] # 이전 에폭 값
        pre_loss = checkpoint['loss'] # 이전 손실 값
        count =  checkpoint['train_no']
        print('load?')
        model.load_state_dict(checkpoint['model_state_dict']) # kobert 모델의 가중치와 편향치를 로드한다.
        print('load?')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # 최적화 함수의 가중치와 편향치를 로드한다.
        print('load?')
        print(f"load pretrain from: {save_ckpt_path}, epoch={pre_epoch}, loss={pre_loss}")
        # best_epoch += 1



    losses =[]
    offset = pre_epoch

    for step in range(n_epoch):
        epoch = step + offset
        count = 0
        loss = train(device, epoch, model, optimizer, train_loader, save_step, save_ckpt_path, train_step)
        losses.append(loss)