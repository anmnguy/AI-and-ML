'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import torch
from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self, model, loss_function, test_loader):
        print('evaluating performance...')

        with torch.no_grad():
            y_true, y_pred, losses = [], [], []
            for X_batch, y_batch in test_loader:
                pred = model(X_batch)
                loss = loss_function(pred, y_batch.long())
                losses.append(loss.item())
                y_true.append(y_batch)
                y_pred.append(pred.argmax(dim=-1))

            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)

        return accuracy_score(y_true.detach().numpy(), y_pred.detach().numpy())
