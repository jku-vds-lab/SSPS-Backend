import tensorflow as tf
from tensorflow.keras.models import Model
import os
import tempfile
import numpy as np

def remove_activation(model, layer_idx = -1):
    model = tf.keras.models.clone_model(model)

    #---last layer should not have an activation for gradCAM!
    layer = model.layers[layer_idx]
    layer.activation = tf.keras.activations.linear
    model = apply_modifications(model)
    
    return model
    
def apply_modifications(model, custom_objects=None):
    """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.
    Args:
        model: The `keras.models.Model` instance.
    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)
        
        
        
def get_mask_for_class(model, image, class_idx=0): # return mask of input image, for those pixel that have an argmax of class=class_idx
    seg_mask = model.predict(image)[0].argmax(-1)
    mask = seg_mask == class_idx
    return mask[np.newaxis,:,:,np.newaxis]
        
def get_explainable_model(model, mask=None, class_idx=0, out_shape=None, output_layer=-1): # specify output layer such that the output before the softmax layer is considered

    if out_shape is None:
        out_shape = [1 if dim is None else dim for dim in model.layers[-1].output_shape]
    if mask is None:
        mask = np.ones(out_shape)
    
    new_model = model #remove_activation(model)
    mask_out = MultiplyConstantMask(mask, name="mask_layer")(new_model.layers[output_layer].output)
    gap_out = tf.keras.layers.GlobalAveragePooling2D()(mask_out)

    mask_model = Model(new_model.input, gap_out)
    
    return mask_model


class MultiplyConstantMask(tf.keras.layers.Layer):
    def __init__(self, mask, *args, **kwargs):
        super(MultiplyConstantMask, self).__init__(*args, **kwargs)
        self.set_mask(mask)
        
    def call(self, inputs):
        return tf.multiply(inputs, self.mask)
    
    def set_mask(self, mask): # if you want to update mask on the fly
        self.mask = tf.Variable(mask, trainable=True, dtype=tf.float32)
        