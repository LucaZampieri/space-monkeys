import keras.backend as K
import numpy as np


class LogLoss(object):
    
    """Logarithmic loss function for probability classifiers in Keras. Once instantiated produces
    a callable object that takes arguments y_true and y_pred (Keras tensors) to calculate the loss. Class weights
    can be used.
    
    Ussage example:
    
    # Compute some class weights if necessary, random for this example
    class_weights = np.random.randn(n_classes)
    
    # Instantiate the object:
    loss = LogLoss(weights=class_weights)
    
    # Instantiate a non-weighted loss to usee as metric, for example
    metric = LogLoss()
    
    # Create Keras model
    net = models.Sequential(...)
    
    # compile the model using the loss function and metric
    net.compile(
        optimizer='adadelta',
        loss=loss
        metrics=metric
    )
    
    # Fit the model
    net.fit(...)
    
    The loss functions can be used without instantiating the object:
    
    # For example
    loss = LogLoss._weighted_log_loss(weights=class_weights)
    
    # Or
    loss = LogLoss._log_loss
    """

    def __init__(weights=None):
        self.weights = weights
    
    # Define the log loss
    def _log_loss(y_true, y_pred)
        
        # Clip input to avoud numerical problems with log function
        y = K.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # return the log loss value
        return K.sum(y_true * K.log(y))
    
    # Define the weighted log loss
    def _weighted_log_loss(weights:np.ndarray):
        
        """Weighted logarithmic loss function
        Params:
        - weights: np.ndarray with shape (n_classes,) or (n_classes, 1)
        Returns:
        - Keras loss function with arguments (y_true, y_pred) (Keras tensors)
        """
    
        # Check weights type and shape
        if weights is np.array:
            if len(weights.shape) == 1:
                # Transform into (n_classes, 1) so it can be transposed
                weights = np.array([weights])
            elif len(weights.shape) > 3:
                raise ValueError('weights shape must be (n_classes,) or (n_classes, 1)')
            else if weights.shape[1] > 1:
                raise ValueError('weights shape must be (n_classes,) or (n_classes, 1)')
        else:
            weights = np.array([weights])
            
        # Transform weights into Keras backend variable
        weights = K.variable(value=weights)
        
        # Define the actual loss function
        def loss(y_true, y_pred):
            
            # Clip input to avoid numerical problems with log function
            y = K.clip(y_pred, 1e-15, 1 - 1e-15)
            
            # Return the weighted log loss value
            return - K.sum(K.dot(y_true * K.log(y), K.transpose(weights))) / K.sum(weights)
            
        # return the loss function (callable)
        return loss
    
    # Method to make thhe class callable
    def __call__(self, y_true, y_pred):
        if self.weights is None:
            return self._log_loss(y_true, y_pred)
        else:
            return self._weighted_log_loss(self.weights)(y_true, y_pred)
