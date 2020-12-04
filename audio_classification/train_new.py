import argparse
import importlib

import csv
import os

import torch

from audio_video_dataset import get_audio_video_dataset
from models.cnn_encoder import Model
#from loss.TripletLoss import VideoMatchingLoss
from loss.ContrastiveLoss import VideoMatchingLoss
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

class Dataset(torch.utils.data.Dataset):
  def __init__(self, dset):
    super(Dataset, self).__init__()
    self.dset = dset
    #class_combs = list(itertools.combinations(class_list, 2)) # all combinations
    #for c1, c2 in class_combs:
    item = []
    total_length = 0
    for i in range(7):
        pts = torch.tensor(np.random.randint(low=0, high=6, size=len(dset[i])))
        pts = torch.where(pts>=i, pts + 1, pts)
        for j in range(len(dset[i])):
            first_class = i
            sec_class = int(pts[j])
            first_item = (first_class, j)
            sec_item = (sec_class, random.randint(0, len(dset[sec_class])-1))
            if random.random() > 0.5:
                item.append((first_item, sec_item, 0))
            else:
                item.append((sec_item, first_item, 1))
            total_length += 1
    
    self.item = item
    self.total_length = total_length


  def __getitem__(self, index):
    label = self.item[index][2]
    first_class, f_item = self.item[index][0]
    sec_class, s_item = self.item[index][1]
    a1, v1 = self.dset[first_class][f_item]
    a2, v2 = self.dset[sec_class][s_item]
    if(label==0):
        return a1, 1, v1, 1-label
    else:
        return a1, 1, v2, 1-label

  def __len__(self):
    return self.total_length

def main(num_epochs, batch_size):
    torch.device(device)

    dataset = get_audio_video_dataset(
        data_directory, max_length_in_seconds=1, pad_and_truncate=True
    )

    #why is there double indexing
    eg_data = dataset[0][0]

    dataset = Dataset(dataset)
    dataset_len = len(dataset)
    train_len = round(dataset_len * 0.8)
    test_len = dataset_len - train_len
    #, Dataset(test_split)

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(train_len), int(test_len)]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )
    train_dataloader_len = len(train_dataloader)
    model = Model(audio_size = eg_data[0].size(), video_size=eg_data[1].size())
    model = model.to(device)
    loss_fn = VideoMatchingLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train loss", "train accuracy", "test loss", "test accuracy"])
        
        for epoch in tqdm(range(num_epochs)):
            model.train()
            train_loss = 0
            train_correct = 0
            for sample_idx, (audio1, audio2, video, target) in tqdm(enumerate(train_dataloader)):
                b = audio1.shape[0]
                optimizer.zero_grad()
                audio1, audio2, video, target = audio1.to(device), audio2, video.to(device), target.to(device)
                audio1_enc, video_enc = model(audio1, video)
                loss, pred = loss_fn(audio1_enc, video_enc, target)
                loss.backward()
                optimizer.step()
                
                train_loss += b * loss.mean().item()
                predicted = pred >= 0.5
                #torch.argmin(pred, dim=1)
                train_correct += (predicted == target).sum().item()
                print(
                    f"{epoch:06d}-[{sample_idx + 1}/{train_dataloader_len}]: {loss.mean().item()} : {train_correct}"
                )

            print(f"Train loss: {train_loss / train_dataset.__len__()}")
            print(f"Train accuracy: {100 * train_correct / train_dataset.__len__()}")

            # Save the model after every epoch (just in case end before num_epochs epochs)
            torch.save(model.state_dict(), f"model_state/{hparams.model}.pt")

            total_length = len(test_dataset)

            model.eval()

            test_loss = 0
            test_correct = 0
            with torch.no_grad():
                for sample_idx, (audio1, audio2, video, target) in enumerate(test_dataloader):
                    b = audio1.shape[0]
                    audio1, audio2, video, target = audio1.to(device), audio2, video.to(device), target.to(device)
                    audio1_enc, video_enc = model(audio1, video)
                    loss, pred = loss_fn(audio1_enc, video_enc, target)
                    test_loss += b * loss.mean().item()
                    predicted = torch.argmin(pred, dim=1)
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
    main(num_epochs=50, batch_size=2)
