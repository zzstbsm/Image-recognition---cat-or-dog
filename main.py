import numpy as np
import matplotlib.pyplot as plt

import sys,os

from lib.network import network

train = False
train_new = False
# train = True
# train_new = True

filename = "model\cat-dog.h5"

# Import data
train_folder = "./data/train/"
test_folder = "./data/test/"

Image_Width=128
Image_Height=128
Image_Channels=3
input_shape=(Image_Width,Image_Height,Image_Channels)
batch_size = 20
                                                                
net = network(train_new,input_shape,filename)
net.load_data(train_folder)
net.set_image_paramenters(Image_Width,Image_Height,Image_Channels,batch_size)

if train:
    
    ep = 0
    print(net.model.summary())
    for i in range(10):
        print("Current epoch: %d" %ep)
        net.train(epochs=1)
        net.save(filename)
        ep += 1
        
else:
    
    from PIL import Image
    import matplotlib.pyplot as plt
    import random
    
    # Load test data
    test_data = os.listdir(test_folder)
    test_data = [test_folder+pic for pic in test_data]
    
    random.shuffle(test_data)
    
    fig = plt.figure()
    
    for i in range(9):
        picture = Image.open(test_data[i])
        picture = picture.resize(size=(Image_Width,Image_Height))
        picture = np.asarray(picture)
        picture_formatted = picture.reshape((1,picture.shape[0],picture.shape[1],picture.shape[2]))
        
        # Test
        category = net.predict(picture_formatted)
        ax = fig.add_subplot(330+1+i)
        ax.imshow(picture)
        if np.argmax(category) == 1:
            title = "dog"
        else:
            title = "cat"
        ax.set_title(title)
        
    plt.show()
