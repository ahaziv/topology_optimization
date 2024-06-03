import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def to_cuda(x):
    if use_gpu:
        x = x.to('cuda')
    return x


def single_batch_test(loader, i):
    images, labels = next(iter(train_loader))
    images = to_cuda(images)
    labels = to_cuda(labels)

    # forward
    outputs = cnn(images)
    loss = criterion(outputs, labels)

    # backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=1)

    # optimize
    optimizer.step()

    print(f'Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
          % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.item()))


class SurfCovNet(nn.Module):
    def __init__(self, inp_size, num_class):
        super(SurfCovNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2, dilation=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(1),
            nn.BatchNorm2d(16))                     # for batchnorm bias isn't necessary
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2, dilation=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16))                     # for batchnorm bias isn't necessary
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 20, kernel_size=5, stride=1, padding=2, dilation=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(1),
            nn.BatchNorm2d(20))                     # for batchnorm bias isn't necessary
        self.layer4 = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2, dilation=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(20))                     # for batchnorm bias isn't necessary
        self.layer5 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32))                     # for batchnorm bias isn't necessary
        self.fc1 = nn.Sequential(
            nn.Linear(32*16*32, 100),
            nn.ELU())
            # nn.Dropout(p),
            # nn.BatchNorm1d(50)
        self.fc2 = nn.Sequential(
            nn.Linear(100, num_class),
            nn.ELU())
        self.fc_dropout = nn.Dropout(p=0.5)
        self.conv_dropout = nn.Dropout(p=0.25)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.conv_dropout(out)
        out = self.layer3(out)
        out = self.conv_dropout(out)
        out = self.layer4(out)
        out = self.conv_dropout(out)
        out = self.layer5(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc_dropout(out)
        out = self.fc2(out)

        return self.logsoftmax(out)



# ---------------------------------------------------------- main ---------------------------------------------------- #
if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    print(f'Use GPU: {use_gpu}')
    flag_calc_norm = False                 # signals if to calculate the normalization and std values for the net
    # HyperParameters
    batch_size = 32
    img_dim = [128, 256, 3]
    num_classes = 6
    num_epochs = 64
    learning_rate = 0.001
    train_data_ratio = 0.8

    cnn = SurfCovNet(img_dim[0] * img_dim[1] * img_dim[2], num_classes)
    print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))
    cnn = to_cuda(cnn)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=1e-4)

    min_error = 10
    for epoch in range(num_epochs):
    #     single_batch_test(train_loader, epoch)

        learning_rate = 0.1 * np.exp(-6.9 * epoch)
        for i, (images, labels) in enumerate(train_loader):
            images = to_cuda(images)
            labels = to_cuda(labels)

            # forward
            outputs = cnn(images)
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=1)

            # optimize
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.item()))
                if loss.item() <= min_error:
                    torch.save(cnn.state_dict(),
                               os.path.join(os.getcwd(), 'trained_models', f'loss_{round(loss.item(), 4)}.pt'))
                    min_error = loss.item()

        if (epoch+1) % 10 == 0:
            cnn.eval()
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = to_cuda(images)
                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum()

            print('Test Accuracy of the model: %d %%' % (100 * correct/total))