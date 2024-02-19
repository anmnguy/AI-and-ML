'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
import torch
from torch.autograd import Variable


class Evaluate_Accuracy(evaluate):
    data = None
    test_loader = None
    model = None
    
    def evaluate(self):
        correct = 0
        for images, labels in self.test_loader:
            images = Variable(images).float()
            output = self.model(images)
            predicted = torch.max(output, 1)[1]
            correct += (predicted == labels).sum()
        accuracy = float(correct) / (len(self.test_loader) * 32)
        return accuracy
