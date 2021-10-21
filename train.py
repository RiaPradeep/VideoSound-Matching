import argparse
import importlib
import random

import csv
from tqdm import tqdm

import torch
import numpy as np

from audio_video_dataset import get_audio_video_dataset
from models.cnn_encoder import Model

torch.backends.cudnn.benchmark = True
data_directory = "raw"

random.seed(10417) 
torch.random.manual_seed(10617) 

class Dataset(torch.utils.data.Dataset):
  def __init__(self, dset, train=False):
    super(Dataset, self).__init__()
    self.dset = dset
    self.train = train
    self.item = []

    self.len_each = 99
    self.train_len = int(99 * 0.8)
    self.test_len = 99 - self.train_len
    self.dset_len = self.train_len if self.train else self.test_len
    self.start_pt = 0 if self.train else self.train_len
    self.total_length = self.dset_len * len(self.dset)
    
    # Generate pairs of samples to train on
    for i in range(len(dset)):
        # Randomly generate second class to be paired with
        pts = torch.tensor(np.random.randint(low=0, high=len(self.dset)-1, size=self.dset_len))
        pts = torch.where(pts>=i, pts + 1, pts)
        
        for j in range(self.dset_len):
            first_class = i
            first_item = (first_class, j+self.start_pt)
            # Every other point is chosen to be similar
            if hparams.loss == "triplet":
                sec_class = int(pts[j])
                label = 1 # expect to predict 1 instead of 2
            elif j % 2 == 0:
                sec_class = int(pts[j])
                label = 0
            else:
                sec_class = i
                label = 1
            sec_item = (sec_class, random.randint(self.start_pt, self.dset_len + self.start_pt -1 ))
            self.item.append((first_item, sec_item, label))

# Returns the relevent data for given index
  def __getitem__(self, index):
    label = self.item[index][2]
    first_class, f_item = self.item[index][0]
    sec_class, s_item = self.item[index][1]
    a1, v1 = self.dset[first_class][f_item]
    a2, v2 = self.dset[sec_class][s_item]
    return a1, a2, v1, v2, label

  def __len__(self):
    return self.total_length

def train(num_epochs, batch_size):
    torch.device(device)

    # Load and set up data
    dataset = get_audio_video_dataset(
        data_directory, max_length_in_seconds=1, pad_and_truncate=True
    )

    audio_size = dataset[0][0][0].size()
    video_size = dataset[0][0][1].size()
    train_dataset = Dataset(dataset, True)
    val_dataset = Dataset(dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=hparams.num_workers, pin_memory=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=hparams.num_workers, pin_memory=False
    )
    train_dataloader_len = len(train_dataloader)
    
    # Set up training structures
    model = Model(audio_size = audio_size, video_size=video_size, loss_type=hparams.loss).to(device)
    loss_fn = Loss().to(device)
    if hparams.optimizer == "Adam": # also default optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    elif hparams.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=hparams.lr, momentum=hparams.momentum)

    with open(f'{hparams.model}_{hparams.loss}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train loss", "train accuracy", "test loss", "test accuracy"])
        
        # Start training
        for epoch in tqdm(range(num_epochs)):
            # Run model on training data
            model.train()
            train_loss = 0
            train_correct = 0
            for sample_idx, (audio1, audio2, video1, video2, target) in tqdm(enumerate(train_dataloader)):
                b = audio1.shape[0]
                optimizer.zero_grad()
                if hparams.loss == "multi":
                    audio1, audio2 = audio1.to(device), audio2.to(device)
                    video1, video2 = video1.to(device), video2.to(device)
                    target = target.to(device)
                elif hparams.loss == "triplet":
                    audio1, audio2, video1, target = audio1.to(device), audio2.to(device), video1.to(device), target.to(device)
                else:
                    audio1, video2, target = audio1.to(device), video2.to(device), target.to(device)
                    
                
                # Run and update model
                if hparams.loss == "bce":
                    audio_enc, video_enc = model(audio1, video2)
                    loss, pred = loss_fn(audio_enc, video_enc, target)
                elif hparams.loss == "cos":
                    pred, enc = model(audio1, video2)
                    loss, pred = loss_fn(pred, enc, target)
                elif hparams.loss == "multi":
                    pred, audio1_enc, audio2_enc, video1_enc, video2_enc = model((audio1, audio2), (video1, video2))
                    loss, pred = loss_fn(pred, audio1_enc, audio2_enc, video1_enc, video2_enc, target)
                elif hparams.loss == "triplet":
                    audio1_enc, audio2_enc, video1_enc = model((audio1, audio2), video1)
                    loss, predicted = loss_fn(audio1_enc, audio2_enc, video1_enc)

                loss.backward()
                optimizer.step()
                
                # Update loss and accuracy
                train_loss += b * loss.mean().item()
                predicted = (pred >= 0.5) * torch.ones(pred.shape).to(device)
                train_correct += (predicted == target).sum().item()
                print(
                    f"{epoch:06d}-[{sample_idx + 1}/{train_dataloader_len}]: {loss.mean().item()} : {train_correct}", flush=True
                )

            print(f"Train loss: {train_loss / train_dataset.__len__()}")
            print(f"Train accuracy: {100 * train_correct / train_dataset.__len__()}")

            # Save the model after every epoch
            torch.save(model.state_dict(), f"model_state/{hparams.model}_{hparams.loss}_{epoch}.pt")

            # Run model on validation data
            model.eval()
            val_loss = 0
            val_correct = 0
            with torch.no_grad():
                for sample_idx, (audio1, audio2, video1, video2, target) in tqdm(enumerate(val_dataloader)):
                    b = audio1.shape[0]
                    if hparams.loss == "multi":
                        audio1, audio2 = audio1.to(device), audio2.to(device)
                        video1, video2 = video1.to(device), video2.to(device)
                        target = target.to(device)
                    elif hparams.loss == "triplet":
                        audio1, audio2, video1, target = audio1.to(device), audio2.to(device), video1.to(device), target.to(device)
                    else:
                        audio1, video2, target = audio1.to(device), video2.to(device), target.to(device)
                        
                    
                    # Run model
                    if hparams.loss == "bce":
                        audio_enc, video_enc = model(audio1, video2)
                        loss, pred = loss_fn(audio_enc, video_enc, target)
                    elif hparams.loss == "cos":
                        pred, enc = model(audio1, video2)
                        loss, pred = loss_fn(pred, enc, target)
                    elif hparams.loss == "multi":
                        pred, audio1_enc, audio2_enc, video1_enc, video2_enc = model((audio1, audio2), (video1, video2))
                        loss, pred = loss_fn(pred, audio1_enc, audio2_enc, video1_enc, video2_enc, target)
                    elif hparams.loss == "triplet":
                        audio1_enc, audio2_enc, video1_enc = model((audio1, audio2), video1)
                        loss, predicted = loss_fn(audio1_enc, audio2_enc, video1_enc)
                    
                    # Update loss and accuracy
                    val_loss += b * loss.mean().item()
                    predicted = (pred >= 0.5) * torch.ones(pred.shape).to(device)
                    val_correct += (predicted == target).sum().item()

                print(f"Evaluation loss: {val_loss / val_dataset.__len__()}")
                print(f"Evaluation accuracy: {100 * val_correct / val_dataset.__len__()}")
            
            # Save this epochs results
            writer.writerow([epoch, (train_loss / train_dataset.__len__()), (100 * train_correct / train_dataset.__len__()),
                                    (val_loss / val_dataset.__len__()), (100 * val_correct / val_dataset.__len__())])


def get_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--model', type=str, default='cnn_encoder')
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--num_workers', type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams = get_arguments()
    Model = importlib.import_module(f"models.{hparams.model}").Model
    Loss = importlib.import_module(f"loss.{hparams.loss}").Loss
    train(num_epochs=hparams.epochs, batch_size=hparams.batch_size)
