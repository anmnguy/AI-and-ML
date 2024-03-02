'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt


class Method_RNN(method, nn.Module):
    data = None
    train_loader = None
    test_loader = None
    # it defines the max rounds to train the model
    max_epoch = 11
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    vocab = None

    # it defines the RNN model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        super(Method_RNN, self).__init__()
        self.vocab = {}

        self.embedding_dim = 32  # Dimension of word embeddings
        self.hidden_dim = 64  # Dimension of hidden state
        self.num_layers = 2  # Number of recurrent layers

        self.embedding_layer = nn.Embedding(num_embeddings=133264, embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(self.hidden_dim, 2)


    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        embeddings = self.embedding_layer(x)
        output, (hidden, _) = self.lstm(embeddings)
        return self.linear(hidden[-1])

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        train_losses = []
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            losses = []
            for X_batch, y_batch in tqdm(self.train_loader):
                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch.long())
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_losses.append(sum(losses))
            accuracy = accuracy_evaluator.evaluate(self, loss_function, self.test_loader)
            print("Loss: {:.3f}".format(torch.tensor(losses).mean()), "Accuracy: {:.3f}".format(accuracy))

        plt.plot(range(self.max_epoch), train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Convergence Plot')
        plt.show()

    def test(self, X):
        y_true, y_pred = [], []
        for X, Y in self.test_loader:
            pred = self.forward(X)
            y_pred.append(pred)
            y_true.append(Y)
        y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)

        return y_true.detach().numpy(), nn.functional.softmax(y_pred, dim=-1).argmax(dim=-1).detach().numpy()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        true_y, pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': true_y}
