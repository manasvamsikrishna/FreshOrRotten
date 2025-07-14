import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from tensorflow.keras.models import Model

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, Activation, GlobalMaxPooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import Adam


dataset_path = "C:/Users/jinka/Music/Projects/fresh and rotten/dataset"
print(os.listdir(dataset_path))

TRAIN_PATH = "C:/Users/jinka/Music/Projects/fresh and rotten/dataset/train"
TEST_PATH = "C:/Users/jinka/Music/Projects/fresh and rotten/dataset/test"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
IMG_SHAPE= (224,224)
 
train_datagen = ImageDataGenerator(rescale=1/255.0,
                                 zoom_range=0.2,
                                 shear_range=0.3,
                                 horizontal_flip=True,
                                 brightness_range=[0.5,1.5])

test_datagen = ImageDataGenerator(rescale=1/255.0)


train_gen = train_datagen.flow_from_directory(TRAIN_PATH,
                                            target_size=IMG_SHAPE,
                                            batch_size=BATCH_SIZE,
                                            class_mode="binary")

test_gen = test_datagen.flow_from_directory(TEST_PATH,
                                            target_size=IMG_SHAPE,
                                            batch_size=BATCH_SIZE,
                                            class_mode="binary")


classes_dict = dict(test_gen.class_indices)

classes_dict = {v: k for k,v in classes_dict.items()}
 
images, labels = next(train_gen)

plt.figure(figsize=(20,10))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(images[i])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(classes_dict[labels[i]])
plt.show()    
    
    

inception = InceptionV3(weights='imagenet',input_shape=(224, 224, 3),include_top=False)
 
inception.summary()

layers = inception.layers
print(f'Number of Layers: {len(layers)}')


TRAIN_SIZE = train_gen.samples
TEST_SIZE = test_gen.samples

callbacks = EarlyStopping(patience = 3, monitor='val_acc')
                        
 
inputs = inception.input

x = inception.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)

x = Dropout(0.5)(x)

outputs = Dense(6, activation ='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)


for layer in layers:
    layer.trainable = False



callbacks = [
    EarlyStopping(monitor='val_accuracy',  # Use 'val_accuracy' for accuracy monitoring
                  mode='max',               # Maximize accuracy
                  patience=5,              # Number of epochs with no improvement before stopping
                  verbose=1)
]


model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen,
    validation_steps=TEST_SIZE // BATCH_SIZE,
    steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
    callbacks=callbacks
)


plt.style.use('ggplot')
plt.figure()
fig,(ax1, ax2)=plt.subplots(1,2,figsize=(19,8))
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.legend(['Training','Validation'])
ax1.set_title('Loss')
ax1.set_xlabel('N. of Epochs')
## plot training accuracy vs validation accuracy 
ax2.plot(history.history['accuracy'])
ax2.plot(history.history['val_accuracy'])
ax2.legend(['Training','Validation'])
ax2.set_title('Acurracy')
ax2.set_xlabel('N.of Epochs')


loss, test_acc = model.evaluate(test_gen)
print("Validation Accuracy = %f \nValidation Loss = %f " % (test_acc, loss))

class_names = list(classes_dict.values())
labels = test_gen.classes
preds =  model.predict(test_gen)
predictions = np.argmax(preds, axis=1)
#show the confusion matrix 
conf_matrix = confusion_matrix(labels, predictions) 
# plot the confusion matrix
fig,ax = plt.subplots(figsize=(12, 10))
sb.heatmap(conf_matrix, annot=True, linewidths=0.01,cmap="magma",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
ax.set_xticklabels(labels = class_names,fontdict=None)
ax.set_yticklabels(labels = class_names,fontdict=None)
plt.show()

test_images,test_labels=next(train_gen)
plt.figure(figsize=(20,15))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(test_images[i])
    plt.xticks([])
    plt.yticks([])
    real = classes_dict[test_labels[i]]
    img = test_images[i].reshape(1,224,224,3)
    predicted = int(np.argmax(model.predict(img),axis=1))
    predicted = classes_dict[predicted]
    plt.xlabel(f"Real: {real}\n Predicted: {predicted}")





