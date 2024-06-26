'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Face_Net(method, nn.Module):
    train_dir = '../../data/stage_3_data/'
    test_dir = '../../data/stage_3_data/'

    data_transform = transforms.Compose([transforms.RandomResizedCrop(256),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.ImageFolder(train_dir, transform=data_transform)
    test_data = datasets.ImageFolder(test_dir, transform=data_transform)
    
    data = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    test_loader = torch.utils.data.DataLoader(train_data, batch_size=32)

    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the CNN model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        super(Face_Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(Kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 46 , 64)
        self.fc2 = nn.Linear(64, 40)
        self.dropout = nn.Dropout(0.2)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 32 * 56 * 46)
        #x = self.dropout(x)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_test(self, X, y):
        model = Face_Net('','')
        # declare optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters())
        loss_function = nn.CrossEntropyLoss()
        train_losses = []
        y_pred = []

        # train the model
        model.train()
        for epoch in range(self.max_epoch):
            correct = 0
            epoch_loss = 0
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                var_X_batch = Variable(X_batch).float()
                var_y_batch = Variable(y_batch)
                optimizer.zero_grad()
                output = model(var_X_batch)
                loss = loss_function(output, var_y_batch)
                loss.backward()
                optimizer.step()

                # total correct predictions
                predicted = torch.max(output.data, 1)[1]
                correct += (predicted == var_y_batch).sum()
                epoch_loss += loss.item()
                y_pred.extend(predicted.tolist())

                if batch_idx % 50 == 0:
                    print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                        epoch, batch_idx * len(X_batch), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader), loss.data.item(),
                               float(correct * 100) / float(32 * (batch_idx + 1))))

            train_losses.append(epoch_loss / len(self.train_loader))

        # plot the training convergence
        plt.plot(range(self.max_epoch), train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Convergence Plot')
        plt.show()

        return y_pred, model

    def run(self):
        print('method running...')
        print('--start training and testing...')
        y_pred, model = self.train_test(self.data['train']['X'], self.data['train']['y'])
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        accuracy_evaluator.model = model
        accuracy_evaluator.test_loader = self.test_loader
        accuracy = accuracy_evaluator.evaluate()
        return {'pred_y': y_pred, 'true_y': self.data['test']['y']}, accuracy
