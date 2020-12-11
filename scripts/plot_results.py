import matplotlib.pyplot as plt
import csv
import argparse

def get_data(resfile):
    with open("results/" + resfile, newline='') as f:
        reader = csv.reader(f)
        next(reader) # skip headers
        epochs = []
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        epoch = 0
        for row in reader:
            print(row)
            epochs.append(epoch)
            train_loss.append(float(row[1]))
            train_acc.append(float(row[2]))
            test_loss.append(float(row[3]))
            test_acc.append(float(row[4]))
            epoch += 1
        return (epochs, train_loss, train_acc, test_loss, test_acc)


def plot_loss(epochs, train_loss, test_loss, outfile):
    plt.title("Training and Testing Loss")
    plt.xlabel("epochs")
    plt.ylabel("average loss")
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, test_loss, label="Testing Loss")
    plt.legend(loc="best")
    plt.savefig(outfile)
    plt.clf()

def plot_acc(epochs, train_acc, test_acc, outfile):
    plt.title("Training and Testing Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, test_acc, label="Testing Accuracy")
    plt.legend(loc="best")
    plt.savefig(outfile)
    plt.clf()

def get_arguments():
    parser = argparse.ArgumentParser(description='Plotting')
    parser.add_argument('--file_name', type=str, default='bce_video_transformer')
    return parser.parse_args()

if __name__ == '__main__':
    hparams = get_arguments()
    epochs, train_loss, train_acc, test_loss, test_acc = get_data(f"results_{hparams.file_name}.csv")
    plot_loss(epochs, train_loss, test_loss, f"images/{hparams.file_name}_loss.png")
    plot_acc(epochs, train_acc, test_acc, f"images/{hparams.file_name}_acc.png")

    