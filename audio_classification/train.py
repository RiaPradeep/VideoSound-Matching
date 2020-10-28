import argparse

import torch

from audio import get_audio_dataset
from model import AudioCNN
import random

torch.backends.cudnn.benchmark = True
data_directory = "raw/"


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
    #print(torch.Tensor(labels, dtype=int))

    random.shuffle(list2)
    self.comb_list = list(zip(list1, list2))


  def __getitem__(self, index):
    id1 = self.comb_list[index][0]
    
    id2 = self.comb_list[index][1]
    label = self.labels[index]

    if(label==0):
        return self.dset1[id1][0][0], self.dset2[id2][0][0], self.dset1[id1][0][1], torch.tensor([1, 0])
    else:
        return self.dset1[id1][0][0], self.dset2[id2][0][0], self.dset2[id2][0][1], torch.tensor([0, 1])

  def __len__(self):
    return self.total_length

def main(num_epochs, batch_size):
    dataset = get_audio_dataset(
        data_directory, max_length_in_seconds=1, pad_and_truncate=True
    )
    
    dataset_length = len(dataset)
    train_length = round(dataset_length * 0.8)
    test_length = dataset_length - train_length
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(train_length), int(test_length)]
    )
    train_dataset = Dataset(train_dataset)
    test_dataset = Dataset(test_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=1
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=5, num_workers=1
    )
    train_dataloader_len = len(train_dataloader)
    test_dataloader_len = len(test_dataloader)
    audio_cnn = AudioCNN(len(dataset.classes)).to(device)
    cross_entropy = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(audio_cnn.parameters())
    for epoch in range(num_epochs):
        audio_cnn.train()
        for sample_idx, (audio1, audio2, video, target) in enumerate(train_dataloader):
            audio_cnn.zero_grad()
            audio1 = audio1.to(device)
            audio2 = audio2.to(device)
            video = video.to(device)
            target = target.to(device)
            output = audio_cnn(audio1, audio2, video)
            loss = cross_entropy(output, target)


            optimizer.step()

            print(
                f"{epoch:06d}-[{sample_idx + 1}/{train_dataloader_len}]: {loss.mean().item()}"
            )

        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for sample_idx, (audio, target) in enumerate(test_dataloader):
                audio, target = audio.to(device), target.to(device)

                output = audio_cnn(audio)
                test_loss += cross_entropy(output, target)

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            print(f"Evaluation loss: {test_loss.mean().item() / test_dataloader_len}")
            print(f"Evaluation accuracy: {100 * correct / total}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(num_epochs=1, batch_size=2)