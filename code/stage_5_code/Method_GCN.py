'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class Layer_GCN(nn.Module):

    def __init__(self, input_dim, output_dim, bias = True):
        super(Layer_GCN, self).__init__()

        self.weight = torch.nn.Parameter(torch.zeros(input_dim, output_dim))
        if bias:
            self.bias = torch.nn.Parameter(torch.ones(output_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj_m):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj_m, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class Method_GCN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 200
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    input_dim = 0
    hidden_dim = 128
    output_dim = 0

    # it defines the RNN model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.gc1 = None
        self.gc2 = None

    # initialize GCN layers with correct dimensions
    def initialize_gcn_layers(self):
        self.gc1 = Layer_GCN(self.input_dim, self.hidden_dim)
        self.gc2 = Layer_GCN(self.hidden_dim, self.output_dim)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = nn.ReLU()(x)
        x = nn.Dropout(0.5)(x)
        x = self.gc2(x, edge_index)
        return x

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        self.initialize_gcn_layers()
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        train_losses = []
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred_train = self.forward(torch.FloatTensor(np.array(X)), self.data['graph']['utility']['A'])[self.data['train_test_val']['idx_train']]
            y_pred_test = self.forward(torch.FloatTensor(np.array(X)), self.data['graph']['utility']['A'])[self.data['train_test_val']['idx_test']]
            y_pred_val = self.forward(torch.FloatTensor(np.array(X)), self.data['graph']['utility']['A'])[self.data['train_test_val']['idx_val']]
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(y[self.data['train_test_val']['idx_train']])
            # calculate the training loss
            train_loss = loss_function(y_pred_train, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            # collect the training loss for plotting
            train_losses.append(train_loss.item())

            if epoch % 10 == 0:
                accuracy_evaluator.data = {'true_y': y[self.data['train_test_val']['idx_val']], 'pred_y': y_pred_val.max(1)[1]}
                accuracy, precision, recall, f1 = accuracy_evaluator.evaluate()
                print('Epoch:', epoch, 'Accuracy:', accuracy, 'Loss:', train_loss.item(), ' Precision:', precision,
                      'Recall:', recall, 'F1 Score:', f1, )

        # plot the training convergence
        plt.plot(range(self.max_epoch), train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Convergence Plot')
        plt.show()

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)), self.data['graph']['utility']['A'])[self.data['train_test_val']['idx_test']]
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['graph']['X'], self.data['graph']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['graph']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['graph']['y'][self.data['train_test_val']['idx_test']]}
