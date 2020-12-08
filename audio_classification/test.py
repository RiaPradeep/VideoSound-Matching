import argparse
import importlib

import csv
import os

import torch

from audio_video_dataset import get_audio_video_dataset
from models.cnn_encoder import Model
from loss.BCELoss import VideoMatchingLoss
import random
import itertools
import numpy as np
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
data_directory = "/work/sbali/WildMix/raw"
# 7 classes - acoustic guitar, bird, cat, child speech, flute, piano, waterfall
test_CLASSES = ["acoustic_guitar", "waterfall", "bird"]
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
  def __init__(self, dset):
    super(Dataset, self).__init__()
    self.dset = dset
    item = []
    self.len_each = 50

    self.dset_len = 50
    self.start_pt = 0
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
    self.total_length = self.dset_len * len(self.dset)


  def __getitem__(self, index):
    label = self.item[index][2]
    first_class, f_item = self.item[index][0]
    sec_class, s_item = self.item[index][1]
    a1, v1 = self.dset[first_class][f_item]
    a2, v2 = self.dset[sec_class][s_item]
    return a1, 1, v2, label


  def __len__(self):
    return self.total_length

def main():
    torch.device(device)
    dataset = get_audio_video_dataset(
        data_directory, max_length_in_seconds=1, pad_and_truncate=True
    )

    eg_data = dataset[0][0]
    test_dataset = Dataset(dataset)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False
    )
    test_dataloader_len = len(test_dataloader)
    model = Model(audio_size = eg_data[0].size(), video_size=eg_data[1].size(), loss_type='bce')
    model = model.to(device)
    checkpt = torch.load(hparams.checkpoint)
    model.load_state_dict(checkpt)
    loss_fn = VideoMatchingLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with open(f'test_bce_{hparams.model}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["test loss", "test accuracy"])
        test_loss = 0
        test_correct = 0
        print(len(test_dataset))
        with torch.no_grad():
            for sample_idx, (audio1, audio2, video, target) in tqdm(enumerate(test_dataloader)):
                b = audio1.shape[0]
                audio1, audio2, video, target = audio1.to(device), audio2, video.to(device), target.to(device)
                audio1_enc, video_enc = model(audio1, video)
                loss, pred = loss_fn(audio1_enc, video_enc, target)
                test_loss += b * loss.mean().item()
                predicted = (pred >= 0.5) * torch.ones(pred.shape).to(device)
                test_correct += (predicted == target).sum().item()
                print(test_correct)
            print(f"Evaluation loss: {test_loss / test_dataset.__len__()}")
            print(f"Evaluation accuracy: {100 * test_correct / test_dataset.__len__()}")
            
        writer.writerow([(test_loss / len(test_dataset)), (100 * test_correct /len(test_dataset))])


def get_arguments():
    parser = argparse.ArgumentParser(description='testing')
    parser.add_argument('--model', type=str, default='video_transformer')

    parser.add_argument('--checkpoint', type=str, default='/work/sbali/VideoSound-Matching/audio_classification/model_state/bce_video_transformer.pt')
    parser.add_argument('--batch_size', type=str, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams = get_arguments()
    Model = importlib.import_module(f"models.{hparams.model}").Model
    batch_size = hparams.batch_size
    main()
