import numpy as np
"""
The cross entropy Loss is often used in classification task, typically in conjunction with SoftMax
(or Sigmoid).
Task:
Implement a class CrossEntropyLoss in the file: “Loss.py” in folder “Optimization”.
When forward propagating we now additionally need the argument label tensor for
forward(prediction tensor, label tensor) and backward(label tensor). We don’t con-
sider the loss function as a layer like the previous ones in our framework, thus it should not
inherit the base layer.
 Write a constructor for this class, receiving no arguments.
 Implement a method forward(prediction tensor, label tensor) which computes the
Loss value according the CrossEntropy Loss formula accumulated over the batch.
 Implement a method backward(label tensor) which returns the error tensor for the
previous layer. The backpropagation starts here, hence no error tensor is needed.
Instead, we need the label tensor.
Hint: the same hint as before applies.
 Remember: Loops are slow in Python. Use NumPy functions instead!
You can verify your implementation using the provided testsuite by providing the commandline
parameter TestCrossEntropyLoss
"""
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