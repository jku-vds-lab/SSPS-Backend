from typing import Any
from flask import Flask, request, jsonify
from flask.sessions import NullSession
import matplotlib
import matplotlib.pyplot as plt
from werkzeug.wrappers import response
import tensorflow as tf
import numpy as np
import pandas as pd
import tf_explain
import importlib
from tensorflow.keras.models import Model
import cv2
import json



# https://github.com/nguyenhoa93/cnn-visualization-keras-tf2/blob/master/src/gradcam.py
class GradCAM:
    def __init__(self, model, layerName=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        
        self.layerName = layerName

        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def compute_heatmap(self, image, upsample_size, classIdx=None, eps=1e-5):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )
        # record operations for automatic differentiation
            
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs)  # preds after softmax
            if classIdx is None:
                classIdx = np.argmax(preds)
            #print(preds.shape)
            loss = preds[:, classIdx]

        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))

        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)

        # Apply reLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, upsample_size,cv2.INTER_LINEAR)

        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        return cam#cam3


def overlay_gradCAM(img, cam3):
    cam3 = np.uint8(255 * cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)

    new_img = 0.3 * cam3 + 0.5 * img

    return (new_img * 255.0 / new_img.max()).astype("uint8")
with tf.device("/cpu:0"):
    #create model
    model = tf.keras.models.load_model("models/mushroom_seg_model.h5")
    print(model.summary())

# Conv_layer list contains  ['','','','','']   the names of the needed layer and it should be added to the on change API 
# as it is '' as a string in line 167 and line  196 as layerName variable,  as the used one and the defult is 'final_conv' and it's already in the list. so i don't take index 
# I take the name itself. 
layer_names=[layer.name for layer in model.layers]
conv_layers=[]
word='conv'
for layer in layer_names:
    if word in layer:
        conv_layers.append(layer)
preprocess_input = tf.keras.applications.vgg16.preprocess_input
pred:any
app = Flask(__name__)
@app.route('/form-example', methods=['GET', 'POST'])
def form_example():
    # handle the POST request
    if request.method == 'POST':
        image = request.files['image']
        image_path= "testimg/" + image.filename
        image.save(image_path)
        with tf.device("/cpu:0"):

            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(384, 384))
            array = tf.keras.preprocessing.image.img_to_array(img)
            # We add a dimension to transform our array into a "batch"
            # of size (1, 299, 299, 3)
            array = preprocess_input(np.expand_dims(array, axis=0))
            pred = model.predict(array)
            print(type(pred[0].argmax(-1).tolist()))
            response = jsonify(pred[0].argmax(-1).tolist())
            header = response.headers
            header['Access-Control-Allow-Origin'] = '*'
            return response

    # otherwise handle the GET request
    response = jsonify(conv_layers)
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/change', methods=['GET', 'POST'])
def change():
    # handle the POST request
    if request.method == 'POST':
        image = request.files['image']
        #print('dsdsdsaww', request.form.get('item'))            
        xval=request.form.get('xselected')
        yval=request.form.get('yselected')
        selected_layer = request.form.get('item2')
        #selected_layer=conv_layers[layer_index]
        print('xval',xval)
        print('yval',yval)

        if xval or yval :
            
            yval=yval.split(',')
            xval=xval.split(',')
            if len(xval) > 1  or len(yval) >1 :
                    

                for i in range(0,len(xval)):
                    xval[i]=int(float(xval[i]))
                    yval[i]=int(float(yval[i]))
                x=np.array(xval).astype(int)
                y=np.array(yval).astype(int)
                xarray=np.arange(x[0],x[1])
                yarray=np.arange(y[0],y[1])
                image_path= "testimg/" + image.filename
                img = tf.keras.preprocessing.image.load_img(image_path, target_size=(384, 384))
                array = tf.keras.preprocessing.image.img_to_array(img)
                # We add a dimension to transform our array into a "batch"
                # of size (1, 299, 299, 3)
                array = preprocess_input(np.expand_dims(array, axis=0))
                pred = model.predict(array)
                
                seg_mask = model.predict(array)[0].argmax(-1)
                for i in range(0,len(xarray)):
                    for j in range(0,len(yarray)):
                        seg_mask[xarray[i]][yarray[j]]=20

                
                mask = seg_mask == 20
                mask = mask[np.newaxis,:,:,np.newaxis]    

                mask_model = tf_explain.get_explainable_model(model, mask=mask)

                with tf.device("/cpu:0"):

                    gcam = GradCAM(model=mask_model, layerName=selected_layer)#"aggregation_conv")#"center_block1_relu")
                    cam = gcam.compute_heatmap(array, upsample_size=(384,384))
                #print(cam.tolist())
                response = jsonify(cam.tolist())
                header = response.headers
                header['Access-Control-Allow-Origin'] = '*'
                #print(response)
                return response  
            else:
                for i in range(0,len(xval)):
                    xval[i]=int(float(xval[i]))
                    yval[i]=int(float(yval[i]))
                x=np.array(xval).astype(int)
                y=np.array(yval).astype(int)
                xarray=x
                yarray=y
                image_path= "testimg/" + image.filename
                img = tf.keras.preprocessing.image.load_img(image_path, target_size=(384, 384))
                array = tf.keras.preprocessing.image.img_to_array(img)
                # We add a dimension to transform our array into a "batch"
                # of size (1, 299, 299, 3)
                array = preprocess_input(np.expand_dims(array, axis=0))
                pred = model.predict(array)
                
                seg_mask = model.predict(array)[0].argmax(-1)
                for i in range(0,len(xarray)):
                    for j in range(0,len(yarray)):
                        seg_mask[xarray[i]][yarray[j]]=20

                
                mask = seg_mask == 20
                mask = mask[np.newaxis,:,:,np.newaxis]    

                mask_model = tf_explain.get_explainable_model(model, mask=mask)

                with tf.device("/cpu:0"):

                    gcam = GradCAM(model=mask_model, layerName=selected_layer)#"aggregation_conv")#"center_block1_relu")
                    cam = gcam.compute_heatmap(array, upsample_size=(384,384))
                #print(cam.tolist())
                response = jsonify(cam.tolist())
                header = response.headers
                header['Access-Control-Allow-Origin'] = '*'
                #print(response)
                return response                  

        else:

            image_path= "testimg/" + image.filename
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(384, 384))
            array = tf.keras.preprocessing.image.img_to_array(img)
            # We add a dimension to transform our array into a "batch"
            # of size (1, 299, 299, 3)
            array = preprocess_input(np.expand_dims(array, axis=0))
            pred = model.predict(array)
            #print(type(pred[0].argmax(-1).tolist()))
            feat_num= pred.shape[3]
            a_list = list(range(0, feat_num))
            feat_names=['background', 'cap','ring', 'stipe', 'gills', 'volva', 'mycelium']
            class_idx = int(request.form.get('item'))
            # get grad-cam for class with idx 1
            # for that we need to generate a mask that masks out all pixels from other classes (in the interactive tool, we would like users to also be able to define this mask, not only derive it from the class)
            mask = tf_explain.get_mask_for_class(model, array, class_idx=class_idx)
            # and we create an explainable model from the segmentation model
            mask_model = tf_explain.get_explainable_model(model, mask=mask)

            with tf.device("/cpu:0"):
                gcam = GradCAM(model=mask_model, layerName=selected_layer)#"aggregation_conv")#"center_block1_relu")
                cam = gcam.compute_heatmap(array, upsample_size=(384,384), classIdx=class_idx)
            #print(cam.tolist())
            response = jsonify(cam.tolist())
            header = response.headers
            header['Access-Control-Allow-Origin'] = '*'
            #print(response)
            return response

    # otherwise handle the GET request
    return '''
           <form method="POST">
               <div><label>Language: <input type="text" name="language"></label></div>
               <div><label>Framework: <input type="text" name="framework"></label></div>
               <input type="submit" value="Submit">
           </form>'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1025, debug= True, use_reloader = True)