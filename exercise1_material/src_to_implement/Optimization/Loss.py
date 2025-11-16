import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        pass

    
    def forward(self, prediction_tensor, label_tensor):
        #formula:  b=1 to B sum(-ln(pred_k + epsilon)) where y_k = 1 (pred_k is prediction for y_k) and epsilon is smallest representable number (np.finfo.eps)
        #I think shape of pred_tensor is (batch_size x #classes); of label_tensor should be the same (in each row exactly one 1, rest 0 (one-hot encoded))

        self._pred_plus_epsilon = prediction_tensor + np.finfo(prediction_tensor.dtype).eps
        transformed_preds = -np.log(self._pred_plus_epsilon)
        
        loss = np.sum(transformed_preds * label_tensor)
        
        return loss

    def backward(self, label_tensor): 
        # Doesn't need an error (the error computation starts here)
        # formula: - y / (pred + epsilon); so we want to see how big the error of prediction was (if pred ~ 1 => very small error etc)

        return -1 * np.divide(label_tensor, self._pred_plus_epsilon)