import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

   
class wMSE:
    """
    Mean squared error.
    Handles weightmap (assumed to be concatenated to y_true on the last axis).
    """
    
    def __init__(self, is_weighted=False, is_stacked=False):
        self.is_weighted = is_weighted 
        self.is_stacked = is_stacked
        
    def loss(self, y_true, y_pred): 
        
        if self.is_weighted:
            if self.is_stacked:
                weights = y_true
                y_true, y_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)
            else:
                y_true, weights = tf.split(y_true, num_or_size_splits=2, axis=-1)
            loss = K.sum(weights * K.square(y_true - y_pred)) / K.sum(weights)        
        else:
            if self.is_stacked:
                y_true, y_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)
            loss = K.mean(K.square(y_true - y_pred))
        
        return loss
        
        
