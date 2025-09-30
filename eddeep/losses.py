import tensorflow.keras.ops as KO


class regul_unidir:
    
    def __init__(self, p=2, order=1):
        self.p = p
        self.order = order
    
    def loss(self, field):
        """
        Regularization loss of a unidir (scalar) field.
        """

        ndims = len(field.shape) - 2
        loss = 0.0
        
        for axis in range(1, ndims + 1):
            
            diff = KO.diff(field, axis=axis)
            loss += KO.mean(KO.abs(diff) ** self.p)

        return loss
