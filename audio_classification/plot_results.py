import matplotlib.pyplot as plt
import csv

def get_data(resfile):
    with open("results/" + resfile, newline='') as f:
        reader = csv.reader(f)
        next(reader) # skip headers
        
        epochs = []
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        for row in reader:
            epochs.append(int(row[0]) + 1)
            train_loss.append(float(row[1]))
            train_acc.append(float(row[2]))
            test_loss.append(float(row[3]))
            test_acc.append(float(row[4]))
    
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

if __name__ == '__main__':
    epochs, train_loss, train_acc, test_loss, test_acc = get_data("results_base_final.csv")

    plot_loss(epochs, train_loss, test_loss, "base_loss.png")
    plot_acc(epochs, train_acc, test_acc, "base_acc.png")

    