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
import torch.nn as nn

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
K = 3
random.seed(10417) 
torch.random.manual_seed(10617) 
'''
# one shot learning
class Dataset(torch.utils.data.Dataset):
  def __init__(self, dset):
    super(Dataset, self).__init__()
    self.dset = dset
    item_classes = []
    item_id = []
    test_val = []
    self.len_each = 10
    self.dset_len = 10
    self.start_pt = 0
    # for each of the evaluation classes
    for i in range(3):
        cur_class = torch.tensor([i])
        for j in range(self.dset_len):
            # select random classes
            all_class_type = [k for k in range(len(self.dset)) if not(k==i)]
            classes = [i for _ in range(K-1)]
            #np.random.choice(all_class_type, size=K-1, replace=False)
            classes = torch.tensor(classes)
            #torch.tensor(np.random.randint(low=0, high=len(self.dset)-1, size=K-1))
            all_classes = torch.cat((cur_class, classes), dim=0) 
            all_item_nos = torch.tensor(np.random.randint(low=self.dset_len, high=40, size=K))
            item_classes.append(all_classes)
            item_id.append(all_item_nos)
            test_val.append((i, j))
    self.total_length = self.dset_len * 3
    self.item_classes = item_classes
    self.item_id = item_id
    self.test_val = test_val


  def __getitem__(self, index):
    classes = self.item_classes[index]
    items = self.item_id[index]
    cur_elem = self.test_val[index]
    #print(classes)
    comp_vals = [self.dset[classes[i].item()][items[i].item()] for i in range(len(classes))]
    cur_val = self.dset[cur_elem[0]][cur_elem[1]]
    return cur_val, comp_vals

  def __len__(self):
    return self.total_length
'''


class Dataset(torch.utils.data.Dataset):
  def __init__(self, dset, train=True):
    super(Dataset, self).__init__()
    self.dset = dset
    self.train = train
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

    label = self.item[index][2]
    print("LABEL", label)
    first_class, f_item = self.item[index][0]
    sec_class, s_item = self.item[index][1]
    a1, v1 = self.dset[first_class][f_item]
    a2, v2 = self.dset[sec_class][s_item]
    return (a1, a2), (v1, v2)
    #(a1, a1), (v1, v2), label


  def __len__(self):
    return self.total_length

class OneShotLearning(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = device

    def forward(self, cur_val, comp_vals):
        cur_audio = cur_val[0]        
        sim = self.model(cur_val[0].to(device), comp_vals[0].to(device))
        #similarities = [self.model(cur_audio.to(device), comp_val[1].to(device))[0] for comp_val in comp_vals]
        #print(sim)
        return sim[0].reshape(-1)
        #torch.tensor(similarities)



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
    model = OneShotLearning(model).to(device)

    loss_fn = VideoMatchingLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with open(f'test_bce_{hparams.model}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["test loss", "test accuracy"])
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for sample_idx, (val, comp_vals) in tqdm(enumerate(test_dataloader)):
                sim_vals = model(val, comp_vals)
                #val = torch.argmax(sim_vals)
                res = torch.ones(sim_vals.shape).to(device) * (sim_vals>=0.5)
                val = 1
                test_correct += (res).sum().item()
                print(sim_vals, test_correct)
            print(f"Evaluation loss: {test_loss / test_dataset.__len__()}")
            print(f"Evaluation accuracy: {100 * test_correct / test_dataset.__len__()}")
            
        writer.writerow([(test_loss / len(test_dataset)), (100 * test_correct /len(test_dataset))])


def get_arguments():
    parser = argparse.ArgumentParser(description='testing')
    parser.add_argument('--model', type=str, default='video_cnn_lstm')
    parser.add_argument('--checkpoint', type=str, default='/work/sbali/VideoSound-Matching/audio_classification/model_state/multi_video_transformer.pt')
    parser.add_argument('--batch_size', type=str, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams = get_arguments()
    Model = importlib.import_module(f"models.{hparams.model}").Model
    batch_size = hparams.batch_size
    main()
