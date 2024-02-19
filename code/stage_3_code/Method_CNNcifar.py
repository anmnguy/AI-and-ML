'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import torchvision.transforms
import matplotlib.pyplot as plt


class Method_CNN(method, nn.Module):

    #convert data to a normalized torch.FloatTensor
    """
    data_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    """
    
    #train_dir = '../../data/stage_3_data/'
    #test_dir = '../../data/stage_3_data/'

    data = None
    train_loader = None
    test_loader = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 0.01

    # it defines the CNN model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        super(Method_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 128, 256)
        self.fc2 = nn.Linear(256, 10)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv3(x), 2))
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        correct_size = x.size(1) * x.size(2) * x.size(3)
        x = x.view(-1, correct_size)  # Replace `3 * 3 * 64` with the correct size
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_(self, X, y):
        model = Method_CNN('','')
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        train_losses = []
        pred_y = []

        # Train the model
        model.train()
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
