'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Method_CNN(method, nn.Module):
    data = None
    train_loader = None
    test_loader = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the CNN model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        super(Method_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dropout4 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 10)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = x.reshape(-1, 64 * 7 * 7)  # Flatten the output for dense layer
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout4(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_(self, X, y):
        model = Method_CNN('','')
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        train_losses = []
        pred_y = []

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        model.train()

        # Train the model
        for epoch in range(self.max_epoch):
            correct = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                var_images = torch.autograd.Variable(images).float()
                var_labels = torch.autograd.Variable(labels)
                optimizer.zero_grad()
                output = model(var_images)
                loss = loss_function(output, var_labels)
                loss.backward()
                optimizer.step()

                # Total correct predictions
                predicted = torch.max(output.data, 1)[1]
                correct += (predicted == var_labels).sum()
                pred_y.extend(predicted.tolist())
                # print(correct)
                if batch_idx % 10000 == 0:
                    print('Epoch:', epoch, 'Loss:', loss.item(), 'Accuracy:', float(correct) / float(1 * (batch_idx + 1)))
            train_losses.append(loss.item())


        pred_y = torch.tensor(pred_y)

        # plot the training convergence
        plt.plot(range(self.max_epoch), train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Convergence Plot')
        plt.show()

        model.eval()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        accuracy_evaluator.test_loader = self.test_loader
        accuracy_evaluator.model = model
        accuracy = accuracy_evaluator.evaluate()

        return pred_y, accuracy

    def run(self):
        print('method running...')
        print('--start training and testing...')
        pred_y, accuracy = self.train_(self.data['train']['X'], self.data['train']['y'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}, accuracy
