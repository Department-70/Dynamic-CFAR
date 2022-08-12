'''
Author: Andrew H. Fagg
'''

import tensorflow as tf 
from tensorflow import keras

class FractionOfVarianceAccountedFor(keras.metrics.Metric):
    '''
    FVAF = 1 - mse / var
    
    Our challenge in implementing standard Metrics is that the data to compute the metrics
    is provided incrementally (through multiple calls to update_state())
    
    For FVAF, we need to keep track of:
    - Number of samples
    - Sum of the squared errors (y-pred)^2
    - Sum of the true values
    - Sum of the squared true values
    
    '''
    
    def __init__(self, ndims, name='fvaf', **kwargs):
        '''
        @param ndims Number of network output dimensions (each dimension is treated
        separately in the FVAF computation)
        '''
        super(FractionOfVarianceAccountedFor, self).__init__(name=name, **kwargs)
        
        # Number of predicted dimensions
        self.ndims = ndims
        
        # Number of samples
        self.N = self.add_weight(name='N', shape=(1,), 
                                 initializer='zeros', dtype=tf.int32)
        
        # Sum of squared true values
        self.sum_squares = self.add_weight(shape=(ndims,), 
                                           name='sum_squares', initializer='zeros',
                                          dtype=tf.float64)
        
        # Sum of true values
        self.sum = self.add_weight(shape=(ndims,),name='sum', initializer='zeros',
                                  dtype=tf.float64)
        
        # Sum of squared errors
        self.sum_squared_errors = self.add_weight(shape=(ndims,),
                                                  name='sum_squared_errors', initializer='zeros',
                                                 dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
        @param y_true Expected output (shape: samples x outputs)
        @param y_pred Predicted output (shape: samples x outputs)
        @param sample_weight Weight of each sample in the performance measure (shape: samples)
        
        TODO: don't yet address sample_weight
        
        NOTE: names help with debugging
        '''
        # Ensure that the inputs are in a standard form
        y_true = tf.cast(y_true, dtype=tf.float64)
        y_pred = tf.cast(y_pred, dtype=tf.float64)
        
        # Sum squared errors
        diff_squared = tf.math.squared_difference(y_true, y_pred, 
                                                  name='diff_squared')
        
        # Sum along examples
        sums = tf.reduce_sum(diff_squared, axis=0, 
                            name='sums1')
        self.sum_squared_errors.assign_add(sums, name='sse_assign')
        
        # Sum
        sums = tf.reduce_sum(y_true, axis=0, name='sums2')
        self.sum.assign_add(sums, name='sum_assign_add')
        
        # Sum squared
        squared = tf.math.square(y_true, name='squared')
        sums = tf.reduce_sum(squared, axis=0, name='sums3')
        self.sum_squares.assign_add(sums, name='sum_squares_assign_add')
        
        # N: just want the first dimension
        dN = tf.slice(tf.shape(y_true), [0], [1], name='slice')
        self.N.assign_add(dN, name='N_assign_add')
        # tf.cast?
        

    def result(self):
        '''
        @return Fvaf for each output dimension (shape: ndims)
        '''
        
        # Total number of samples
        N = tf.cast(self.N, dtype=tf.float64)
        # Mean of true values
        mean = self.sum / N
        # Variance of true valeus
        variance = self.sum_squares / N - tf.square(mean)
        # FVAF
        fvaf = 1.0 - (self.sum_squared_errors / N) / variance
        
        return fvaf
        
    def reset_states(self):
        '''
        Reset the state of the accumulator variables
        
        This is called between epochs and data sets
        '''
        
        self.N.assign(tf.zeros(shape=(1,), dtype=tf.int32))
        self.sum_squares.assign(tf.zeros(shape=(self.ndims), dtype=tf.float64))
        self.sum.assign(tf.zeros(shape=(self.ndims), dtype=tf.float64))
        self.sum_squared_errors.assign(tf.zeros(shape=(self.ndims), dtype=tf.float64))
        
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "ndims": self.ndims}