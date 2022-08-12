'''
Binary-style metrics that allow the true labels to be a float
'''


import tensorflow as tf
from tensorflow import keras


class MyBinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    '''
    This class does the job of BinaryAccuracy except that y_true does not need to be binary
    - All the machinery of BinaryAccuracy is used, but y_true is binarized first using the specified threshold
    '''
    def __init__(self, name='binary_accuracy', dtype=None, threshold=0.5):
        super().__init__(name=name, dtype=dtype, threshold=threshold)
        self.threshold=threshold
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state((y_true>=self.threshold), y_pred, sample_weight)

class MyAUC(tf.keras.metrics.AUC):
    '''
    This class does the job of AUC except that y_true does not need to be binary
    - All the machinery of AUC is used, but y_true is binarized first using the specified threshold
    '''
    def __init__(self, name='AUC', dtype=None, threshold=0.5):
        super().__init__(name=name, dtype=dtype)
        self.threshold=threshold
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state((y_true>=self.threshold), y_pred, sample_weight)