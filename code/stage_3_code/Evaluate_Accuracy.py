'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
import torch


class Evaluate_Accuracy(evaluate):
    data = None
    test_loader = None
    model = None
    
    def evaluate(self):
        correct = 0
        total = 0
        for batch_idx, (test_images, test_labels) in enumerate(self.test_loader):
            test_images = torch.autograd.Variable(test_images).float()
            output = self.model(test_images)
            predicted = torch.max(output, 1)[1]
            correct += (predicted == test_labels).sum().item()
            total += test_labels.size(0)

        return correct / total
