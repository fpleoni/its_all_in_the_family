#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
from collections import defaultdict
import cv2
from glob import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.math import sqrt, square, reduce_sum, sigmoid
from random import choice, sample
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


# In[2]:


# File paths
train_file_path = "../data/train/"
train_relationships_path = "../csv_files/train_relationships.csv"
validation_path = "F09"


# In[3]:


# Functions by Youness Mansar https://github.com/CVxTz/kinship_prediction
all_images = glob(train_file_path + "*/*/*.jpg")

train_images = [x for x in all_images if validation_path not in x]
val_images = [x for x in all_images if validation_path in x]

train_person_to_images_map = defaultdict(list)

ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

for x in train_images:
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

val_person_to_images_map = defaultdict(list)

for x in val_images:
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

relationships = pd.read_csv(train_relationships_path)
relationships = list(zip(relationships.p1.values, relationships.p2.values))
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

train = [x for x in relationships if validation_path not in x[0]]
val = [x for x in relationships if validation_path in x[0]]


def read_img(path):
    img = preprocessing.image.load_img(path, target_size=(160, 160))
    img = preprocessing.image.img_to_array(img)
    img = preprocess_input(img)
    return img


def gen(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x) for x in X2])

        yield [X1, X2], labels


# In[4]:


def euclidean_distance(embedding_1, embedding_2):
    difference = embedding_1 - embedding_2
    sum_of_squares = np.sum(np.square(difference))
    euclidean_distance = sqrt(sum_of_squares)
    return euclidean_distance


# In[ ]:


def contrastive_loss(y_true, y_pred):
    # Contrastive Loss from Hadsell-et-al. 2006.
    # http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_pred) * K.square(K.maximum(margin - y_pred, 0)))


# In[5]:


# Import FaceNet model and FaceNet weights
facenet_model = models.load_model("../facenet/facenet_keras.h5")
facenet_model.load_weights("../facenet/facenet_keras_weights.h5")


# In[6]:


# Define Model Architecture 
def siamese_model():
    left_image = layers.Input(shape = (160, 160, 3))
    right_image = layers.Input(shape = (160, 160, 3))

    model = models.Sequential()
    model.add(facenet_model)
    model.add(layers.Dense(128, activation = "relu"))
    
    x1 = model(left_image)
    x2 = model(right_image)
    
    L2_normalized_layer_1 = layers.Lambda(lambda x: K.l2_normalize(x, axis = 1))
    X1_normal = L2_normalized_layer_1(x1)
    X2_normal = L2_normalized_layer_1(x2)
    
    Eu_distance = tf.convert_to_tensor(euclidean_distance(X1_normal, X2_normal)
    
    prediction = sigmoid(Eu_distance)
    
    siamese_net = models.Model(inputs = [left_image, right_image], outputs = prediction)

    siamese_net.compile(loss = contrastive_loss, metrics = ["acc"], optimizer = optimizers.Adam(0.0001))
    
    return siamese_net


# In[9]:


tensorboard = TensorBoard(log_dir = "./logs_facenet", batch_size = 100, write_images = True,
                                   write_grads = True, write_graph = False)

early_stop = EarlyStopping(monitor = "val_acc", mode = "max", patience = 20)

model_checkpoint = ModelCheckpoint("best_kinship_model_2.h5", monitor = "val_acc", 
                                             mode = "max", save_best_only = True)

reduce_lr_on_plateau = ReduceLROnPlateau(monitor = "val_acc", mode = "max", patience = 10, factor = 0.1)


# In[10]:


kinship_model = siamese_model()
kinship_model.fit_generator(gen(train, train_person_to_images_map, batch_size = 100),
                    validation_data = gen(val, val_person_to_images_map, batch_size = 100), epochs = 200, verbose = 2,
                steps_per_epoch = 200, validation_steps = 100, 
                            callbacks = [tensorboard, early_stop, model_checkpoint, reduce_lr_on_plateau])


# In[ ]:


kinship_model_json = kinship_model.to_json()
with open("kinship_model_facenet_2.json", "w") as json_file:
    json_file.write(kinship_model_json)
    
kinship_model.save_weights("kinship_model_weights_facenet_2.h5")


# In[ ]:




