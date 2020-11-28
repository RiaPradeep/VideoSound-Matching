import argparse

import csv
import os

import torch

from audio import get_audio_dataset
# from model import AudioCNN
from models.cnn_encoder import AudioCNN
from loss.ContrastiveLoss import VideoMatchingLoss
import random

torch.backends.cudnn.benchmark = True
data_directory = "raw"
# 7 classes - acoustic guitar, bird, cat, child speech, flute, piano, waterfall
TRAIN_CLASSES = ["acoustic_guitar", "waterfall", "bird"]
TEST_CLASSES = ["flute", "child_speech"]
VAL_CLASSES = ["cat", "piano"]


class Dataset(torch.utils.data.Dataset):
  def __init__(self, dset):
    super(Dataset, self).__init__()
    self.dset1 =  dset
    self.dset2 =  dset
    print(dset.__len__())
    total_length = dset.__len__()
    list1 = list(range(total_length))
    list2 = list(range(total_length))
    self.labels = (torch.rand(total_length) < 0.5).type(torch.IntTensor)
    self.total_length = total_length
    random.shuffle(list2)
    self.comb_list = list(zip(list1, list2))

  def __getitem__(self, index):
    id1 = self.comb_list[index][0]
    id2 = self.comb_list[index][1]
    label = self.labels[index]

    if(label==0):
        return self.dset1[id1][0][0], self.dset2[id2][0][0], self.dset1[id1][0][1], torch.tensor(0)
    else:
        return self.dset1[id1][0][0], self.dset2[id2][0][0], self.dset2[id2][0][1], torch.tensor(1)

  def __len__(self):
    return self.total_length

def main(num_epochs, batch_size):
    torch.device(device)

    train_dataset =  get_audio_dataset(
        data_directory, d_type="train", max_length_in_seconds=1, pad_and_truncate=True
    )
    val_dataset =  get_audio_dataset(
        data_directory, d_type="val", max_length_in_seconds=1, pad_and_truncate=True
    )
    test_dataset =  get_audio_dataset(
        data_directory, d_type="test", max_length_in_seconds=1, pad_and_truncate=True
    )

    train_dataset = Dataset(train_dataset)
    val_dataset = Dataset(val_dataset)
    test_dataset = Dataset(test_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    train_dataloader_len = len(train_dataloader)
    audio_cnn = AudioCNN()
    audio_cnn = audio_cnn.to(device)
    loss_fn = VideoMatchingLoss()
    optimizer = torch.optim.Adam(audio_cnn.parameters())

    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train loss", "train accuracy", "validation loss", "validation accuracy", "test loss", "test accuracy"])
        for epoch in range(num_epochs):
            audio_cnn.train()
            train_loss = 0
            train_correct = 0
            for sample_idx, (audio1, audio2, video, target) in enumerate(train_dataloader):
                b = audio1.shape[0]
                optimizer.zero_grad()
                audio1, audio2, video, target = audio1.to(device), audio2.to(device), video.to(device), target.to(device)
                audio1_enc, audio2_enc, video_enc = audio_cnn(audio1, audio2, video)
                loss, pred = loss_fn(audio1_enc, audio2_enc, video_enc, target)
                loss.backward()
                optimizer.step()
                
                train_loss += b * loss.mean().item()
                predicted = torch.argmin(pred, dim=1)
                train_correct += (predicted == target).sum().item()

                print(
                    f"{epoch:06d}-[{sample_idx + 1}/{train_dataloader_len}]: {loss.mean().item()} : {train_correct}"
                )

            print(f"Train loss: {train_loss / train_dataset.__len__()}")
            print(f"Train accuracy: {100 * train_correct / train_dataset.__len__()}")

            # Save the model after every epoch (just in case end before 100 epochs)
            torch.save(audio_cnn.state_dict(), "model_state/encoder.pt")

            audio_cnn.eval()
            
            val_loss = 0
            val_correct = 0

            with torch.no_grad():
                for sample_idx, (audio1, audio2, video, target) in enumerate(val_dataloader):
                    b = audio1.shape[0]
                    audio1, audio2, video, target = audio1.to(device), audio2.to(device), video.to(device), target.to(device)
                    audio1_enc, audio2_enc, video_enc = audio_cnn(audio1, audio2, video)
                    loss, pred = loss_fn(audio1_enc, audio2_enc, video_enc, target)
                    val_loss += b * loss.mean().item()
                    predicted = torch.argmin(pred, dim=1)
                    val_correct += (predicted == target).sum().item()

                print(f"Validation loss: {val_loss / val_dataset.__len__()}")
                print(f"Validation accuracy: {100 * val_correct / val_dataset.__len__()}")

            test_loss = 0
            test_correct = 0
            with torch.no_grad():
                for sample_idx, (audio1, audio2, video, target) in enumerate(test_dataloader):
                    b = audio1.shape[0]
                    audio1, audio2, video, target = audio1.to(device), audio2.to(device), video.to(device), target.to(device)
                    audio1_enc, audio2_enc, video_enc = audio_cnn(audio1, audio2, video)
                    loss, pred = loss_fn(audio1_enc, audio2_enc, video_enc, target)
                    test_loss += b * loss.mean().item()
                    predicted = torch.argmin(pred, dim=1)
                    test_correct += (predicted == target).sum().item()

                print(f"Evaluation loss: {test_loss / test_dataset.__len__()}")
                print(f"Evaluation accuracy: {100 * test_correct / test_dataset.__len__()}")
            
            writer.writerow([epoch, (train_loss / train_dataset.__len__()), (100 * train_correct / train_dataset.__len__()), 
                                    (val_loss / val_dataset.__len__()), (100 * val_correct / val_dataset.__len__()),
                                    (test_loss / test_dataset.__len__()), (100 * test_correct / test_dataset.__len__())])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(num_epochs=100, batch_size=16)
