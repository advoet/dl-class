import numpy as np

from .loss_layer import LossLayer


class SoftmaxCrossEntropyLossLayer(LossLayer):
    def __init__(self, reduction="mean", parent=None):
        """

        :param reduction: mean reduction indicates the results should be summed and scaled by the size of the input (excluding the axis dimension).
            sum reduction means the results should be summed.
        """
        self.reduction = reduction
        self.softmax = None
        self.targets = None
        super(SoftmaxCrossEntropyLossLayer, self).__init__(parent)

    def forward(self, logits, targets, axis=-1): #-> float:
        """

        :param logits: ND non-softmaxed outputs. All dimensions (after removing the "axis" dimension) should have the same length as targets
            2D - (batch) x (class)
        :param targets: (N-1)D class id intege
            1D - (batch)
        :param axis: Dimension over which to run the Softmax and compare labels.
        :return: single float of the loss.
        """
        self.targets = targets

        normalized_logits = logits - logits.max(axis = axis, keepdims = True)
        sum_exp_logits = np.sum(np.exp(normalized_logits), axis = axis, keepdims = True)
        self.softmax = np.exp(normalized_logits)/sum_exp_logits
        log_softmax =  normalized_logits - np.log(sum_exp_logits)

        # targets[k] = integer representing the index of the actual class for kth trial in batch
        # this has the effect of grabbing [k, targets[k]]
        cross_entropy = -log_softmax[np.arange(len(targets)), targets]

        if (self.reduction == "mean"):
            return np.mean(cross_entropy)
        elif (self.reduction == "sum"):
            return np.sum(cross_entropy)
        else:
            return 0


    def backward(self): # -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        gradients = (1/20.)*self.softmax.copy() #multiply this by scaling factor
        gradients[np.arange(len(self.targets)), self.targets] -= 1
        return gradients
