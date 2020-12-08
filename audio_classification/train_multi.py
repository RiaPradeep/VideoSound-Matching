import argparse
import importlib

import csv
import os

import torch

from audio_video_dataset import get_audio_video_dataset
from models.cnn_encoder import Model
#from loss.TripletLoss import VideoMatchingLoss
from loss.MultiLoss import VideoMatchingLoss
import random
import itertools
import numpy as np
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
data_directory = "/work/sbali/WildMix/raw"
# 7 classes - acoustic guitar, bird, cat, child speech, flute, piano, waterfall
TRAIN_CLASSES = ["acoustic_guitar", "waterfall", "bird"]
TEST_CLASSES = ["flute", "child_speech"]
VAL_CLASSES = ["cat", "piano"]

"""
guitar - 101
bird - 100
cat - 50
child speech - 100
flute - 102
piano - 99
waterfall - 75
"""
random.seed(10417) 
torch.random.manual_seed(10617) 

class Dataset(torch.utils.data.Dataset):
  def __init__(self, dset, train=False):
    super(Dataset, self).__init__()
    self.dset = dset
    self.train = train
    #class_combs = list(itertools.combinations(class_list, 2)) # all combinations
    #for c1, c2 in class_combs:
    item = []
    self.len_each = 99
    self.train_len = int( 99 * 0.8 )
    self.test_len = 99 - self.train_len
    self.dset_len = self.train_len if train else self.test_len
    self.start_pt = 0 if train else self.train_len
    if True:
        for i in range(len(dset)):
            pts = torch.tensor(np.random.randint(low=0, high=len(self.dset)-1, size=self.dset_len))
            pts = torch.where(pts>=i, pts + 1, pts)
            for j in range(self.dset_len):
                first_class = i
                first_item = (first_class, j)
                if j % 2==0:
                    sec_class = int(pts[j])
                    sim = 0
                else:
                    sec_class = i
                    sim = 1
                sec_item = (sec_class, random.randint(self.start_pt, self.dset_len + self.start_pt -1 ))
                item.append((first_item, sec_item, sim))
    
    self.item = item
    cur_len = self.train_len if self.train else self.test_len
    self.total_length = self.dset_len * len(self.dset)
    print(self.train, cur_len, len(self.dset))


  def __getitem__(self, index):
    if False:
        cur_class = index //self.len_each
        cur_id = index % self.len_each
        first_item = self.dset[cur_class][cur_id]
        pt = torch.tensor(np.random.randint(low=0, high=len(self.dset)-1, size=1))
        pt = torch.where(pt>=cur_class, pt + 1, pt)
        if cur_id%2 == 0:
            sec_class = int(pt[0])
            label = 0
        else:
            sec_class = cur_class
            label = 1
        sec_item_num = random.randint(0, self.train_len)
        a1, v1 = first_item
        a2, v2 = self.dset[sec_class][sec_item_num]
    else:
        label = self.item[index][2]
        first_class, f_item = self.item[index][0]
        sec_class, s_item = self.item[index][1]
        a1, v1 = self.dset[first_class][f_item]
        a2, v2 = self.dset[sec_class][s_item]
    return (a1, a1), (v1, v2), label


  def __len__(self):
    return self.total_length

def main(num_epochs, batch_size):
    torch.device(device)

    dataset = get_audio_video_dataset(
        data_directory, max_length_in_seconds=1, pad_and_truncate=True
    )

    #why is there double indexing
    eg_data = dataset[0][0]
    '''
    dataset = Dataset(dataset)
    dataset_len = len(dataset)
    train_len = round(dataset_len * 0.8)
    test_len = dataset_len - train_len
    '''
    #, Dataset(test_split)
    train_dataset = Dataset(dataset, True)
    test_dataset = Dataset(dataset)

    '''
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(train_len), int(test_len)]
    )
    '''

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False
    )
    train_dataloader_len = len(train_dataloader)
    model = Model(audio_size = eg_data[0].size(), video_size=eg_data[1].size(), loss_type='multi')
    model = model.to(device)
    loss_fn = VideoMatchingLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with open(f'multi_{hparams.model}results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train loss", "train accuracy", "test loss", "test accuracy"])
        
        for epoch in tqdm(range(num_epochs)):
            model.train()
            train_loss = 0
            train_correct = 0
            for sample_idx, (audio, video, target) in tqdm(enumerate(train_dataloader)):
                b = audio[0].shape[0]
                optimizer.zero_grad()
                audio[0], audio[1] = audio[0].to(device), audio[1].to(device)
                video[0], video[1] = video[0].to(device), video[1].to(device)
                target = target.to(device)
                pred, audio1_enc, audio2_enc, video1_enc, video2_enc = model(audio, video)
                #pred, audio1_enc, audio2_enc, video1_enc, video2_enc, label
                loss, predicted = loss_fn(pred, audio1_enc, audio2_enc, video1_enc, video2_enc, target)
                loss.backward()
                optimizer.step()
                train_loss += b * loss.mean().item()
                #pred = pred.cpu()
                
                predicted = (predicted >= 0.5) * torch.ones(predicted.shape).to(device)
                
                train_correct += (predicted == target).sum().item()
                print(
                    f"{epoch:06d}-[{sample_idx + 1}/{train_dataloader_len}]: {loss.mean().item()} : {train_correct}", flush=True
                )

            print(f"Train loss: {train_loss / train_dataset.__len__()}")
            print(f"Train accuracy: {100 * train_correct / train_dataset.__len__()}")

            # Save the model after every epoch (just in case end before num_epochs epochs)
            torch.save(model.state_dict(), f"/work/sbali/VideoSound-Matching/audio_classification/model_state/mult_{hparams.model}.pt")

            total_length = len(test_dataset)

            model.eval()

            test_loss = 0
            test_correct = 0
            with torch.no_grad():
                for sample_idx, (audio, video, target) in tqdm(enumerate(test_dataloader)):
                    b = audio[0].shape[0]
                    audio[0], audio[1] = audio[0].to(device), audio[1].to(device)
                    video[0], video[1] = video[0].to(device), video[1].to(device)
                    target = target.to(device)
                    pred, audio1_enc, audio2_enc, video1_enc, video2_enc = model(audio, video)
                    #pred, audio1_enc, audio2_enc, video1_enc, video2_enc, label
                    loss, predicted = loss_fn(pred, audio1_enc, audio2_enc, video1_enc, video2_enc, target)
                    test_loss += b * loss.mean().item()
                    print(predicted.shape,  test_dataset.__len__())
                    predicted = (predicted >= 0.5) * torch.ones(predicted.shape).to(device)

                    test_correct += (predicted == target).sum().item()

                print(f"Evaluation loss: {test_loss / test_dataset.__len__()}")
                print(f"Evaluation accuracy: {100 * test_correct / test_dataset.__len__()}")
            
            writer.writerow([epoch, (train_loss / train_dataset.__len__()), (100 * train_correct / train_dataset.__len__()),
                                    (test_loss / test_dataset.__len__()), (100 * test_correct / test_dataset.__len__())])


def get_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--model', type=str, default='cnn_encoder')
    return parser.parse_args()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams = get_arguments()
    Model = importlib.import_module(f"models.{hparams.model}").Model
    main(num_epochs=25, batch_size=2)
