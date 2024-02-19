'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    train_loader = None
    test_loader = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the CNN model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        super(Method_CNN_CIFAR, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 500),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(500, 10),
        )

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        return self.model(x)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_test(self, X, y):
        model = Method_CNN_CIFAR('', '')
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
        evaluations = accuracy_evaluator.evaluate()
        return {'pred_y': y_pred, 'true_y': self.data['test']['y']}, evaluations
