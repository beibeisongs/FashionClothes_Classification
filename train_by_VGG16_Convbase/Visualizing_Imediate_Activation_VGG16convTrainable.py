# encoding=utf-8
# Date: 2018-10-12
# Author: MJUZY


from keras.models import load_model


model = load_model('VGG16FineTune_8_Clothes_Classes.h5')
model.summary()

# img_path = "Skirt.330.jpg"
# img_path = "Tee.15.jpg"
# img_path = "Tee.24908.jpg"
# img_path = "4.jpg"
# img_path = "Shorts.14.jpg"
# img_path = "Shorts.1.jpg"
# img_path = "Sweater.25.jpg"
# img_path = "Dress.7.jpg"
img_path = "Skirt.64.jpg"


from keras.preprocessing import image
import numpy as np


img = image.load_img(img_path, target_size=(224, 224))  # img = <PIL.Image.Image image mode=RGB size=150x150 at 0x23EA12E8A58> _size = <class 'tuple'>: (150, 150)
img_tensor = image.img_to_array(img)    # shape = <class 'tuple'>: (224, 224, 3)
""" Description : of np.expand_dims

    For example, originally shape = (2,3), and axis=0
        then shape changed into (1,2,3)
            when axis = 1then shape changed into (2,1,3)
"""
img_tensor = np.expand_dims(img_tensor, axis=0) # shape = <class 'tuple'>: (1, 224, 224, 3)
img_tensor /= 255.
print(img_tensor.shape) # output: (1, 224, 224, 3)

result = model.predict(img_tensor)
result_array0 = result[0]

result_names = ["长袖上衣", "长裙", "外套", "长裤", "短裤", "短裙", "毛衣", "短袖衬衫"]

""" Notes:

    The id of 上衣：0， 2， 6， 7
    The id of 下衣：1， 3， 4， 5
"""
id_array = [0, 2, 6, 7]

biggest_i = 0
biggest_value = result_array0[id_array[0]]
for i in id_array:
    if result_array0[i] > biggest_value:
        biggest_value = result_array0[i]
        biggest_i = i

print("The result is : ", result_names[biggest_i])

id_array = [1, 3, 4, 5]

biggest_i = 1
biggest_value = result_array0[1]
for i in id_array:
    if result_array0[i] > biggest_value:
        biggest_value = result_array0[i]
        biggest_i = i

print("The result is : ", result_names[biggest_i])


biggest_i = 0
biggest_value = result_array0[0]

for i in range(1, len(result_array0)):
    if result_array0[i] > biggest_value:
        biggest_value = result_array0[i]
        biggest_i = i

print("The best result is : ", result_names[biggest_i])
print("The result array : ", result_array0)


import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])   # because the img_tensor's shape is (1, 224, 224, 3)
plt.show()


print("The length of the model.layers : ", len(model.layers))   # output: 23


from keras import models


# Extracts the outputs of the top eight layers
layer_outputs = [layer.output for layer in model.layers[: 22]]

# Create a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Returns a list of five numpy arrays :
# one array per layer activation
activations = activation_model.predict(img_tensor)

def showActivation(layer_th, filter_th):
    th4_layer_activation = activations[layer_th]
    print("The " + str(layer_th) + " and the " + str(filter_th) + " : ",th4_layer_activation.shape)

    # visualizing the fourth pooling result of the filter's sliding result
    plt.matshow(th4_layer_activation[0, :, :, filter_th], cmap='viridis')
    plt.show()

showActivation(0, 0)
showActivation(0, 1)
showActivation(0, 2)

showActivation(1, 0)
showActivation(1, 1)
showActivation(1, 2)

showActivation(2, 0)
showActivation(2, 1)
showActivation(2, 2)

showActivation(3, 0)
showActivation(3, 1)
showActivation(3, 2)

showActivation(4, 0)
showActivation(4, 1)
showActivation(4, 2)

showActivation(5, 0)
showActivation(5, 1)
showActivation(5, 2)

showActivation(6, 0)
showActivation(6, 1)
showActivation(6, 2)

showActivation(7, 0)
showActivation(7, 1)
showActivation(7, 2)

showActivation(8, 0)
showActivation(8, 1)
showActivation(8, 2)

showActivation(9, 0)
showActivation(9, 1)
showActivation(9, 2)

showActivation(10, 0)
showActivation(10, 1)
showActivation(10, 2)

showActivation(11, 0)
showActivation(11, 1)   # Attention:
                        #   This is an important pattern sensing
                        #   The 11 and the 1 :  (1, 28, 28, 512)
showActivation(11, 2)

showActivation(12, 0)
showActivation(12, 1)
showActivation(12, 2)   # Attention:
                        #   This is an important pattern sensing
                        #   The 12 and the 2 :  (1, 28, 28, 512)

showActivation(13, 0)
showActivation(13, 1)
showActivation(13, 2)

showActivation(14, 0)
showActivation(14, 1)
showActivation(14, 2)

showActivation(15, 0)
showActivation(15, 1)   # Attention:
                        #   This is an important pattern sensing
                        #   The 15 and the 1 :  (1, 14, 14, 512)
showActivation(15, 2)

showActivation(16, 0)
showActivation(16, 1)
showActivation(16, 2)

showActivation(17, 0)
showActivation(17, 1)
showActivation(17, 2)

""" Appendix : 

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              25691136  
_________________________________________________________________
dense_2 (Dense)              (None, 128)               131200    
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 1032      
=================================================================
Total params: 40,538,056
Trainable params: 32,902,792
Non-trainable params: 7,635,264
_________________________________________________________________

"""