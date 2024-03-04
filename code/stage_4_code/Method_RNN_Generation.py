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
from collections import Counter
from Dataset_Loader_Gen import Dataset_Loader

class TokenMapping():
    def __init__(self, text_list, not_found_token: str = 'TOKEN_NOT_FOUND', not_found_id: int | None = None ):
        self.counter = Counter(text_list)
        self.n_tokens: int = len(self.counter) + 1
        self._token2id = {
            token: idx
            for idx, (token, _) in enumerate(self.counter.items())
        }
        self._id2token = {
            idx: token
            for token, idx in self._token2id.items()
        }
        self._not_found_token = not_found_token
        if not_found_id is None:
            self._not_found_id = max(self._token2id.values()) + 1
        else:
            self._not_found_id = not_found_id

        def token2id(self, token: str):
            return self._token2id.get(token, self._not_found_id)
    
        def id2token(self, idx: int):
            return self._id2token.get(idx, self._not_found_token)
    
    def encode(self, text_list):
        '''Encodes list of tokens (strings) into list of IDs (integers)'''
        encoded = [
            self.token2id(token)
            for token in text_list
        ]
        if self._not_found_id not in encoded:
            encoded += [self._not_found_id]
        return encoded

class JokeDataset(dataset):
    def __init__(self, encoded_text, sequence_length):
        self.encoded_text = encoded_text
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.encoded_text) - self.sequence_length

    def __getitem__(self, index):
        x = torch.tensor(self.encoded_text[index: (index+self.sequence_length)], dtype=torch.long)
        y = torch.tensor(self.encoded_text[(index+1): (index+self.sequence_length+1)], dtype=torch.long)
        return x, y


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

        self.embedding_dim = 16  # Dimension of word embeddings
        self.hidden_dim = 32  # Dimension of hidden state
        self.num_layers = 2  # Number of recurrent layers
        self.n_tokens = len(Dataset_Loader.tokenizer()) + 1

        self.embedding_layer = nn.Embedding(num_tokens=self.n_tokens, embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(self.hidden_dim, self.n_tokens)

    def generate_text_by_char():
        sequence, tokenized_text = Dataset_Loader.load()
        generated_tokens = []
        for i in range(len(sequence)):
            new_char = sequence[i]
            generated_tokens.append(new_char)

        full_text = ''.join(tokenized_text + generated_tokens)
        return full_text

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

        gen_output = generate_text_by_char()
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
