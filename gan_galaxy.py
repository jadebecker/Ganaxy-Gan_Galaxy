import keras
import argparse
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, Input, BatchNormalization, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, Adam
import h5py
import numpy as np
from keras.datasets import mnist
import PIL
import pickle
from tensorflow.python.lib.io import file_io
from glob import glob

################### parse arguments ##################################### 
parser = argparse.ArgumentParser()
parser.add_argument( '--job-dir')
parser.add_argument('--img_arr_path', metavar='imgarr', type=str,
                    help='Path to the image array')
argsdeouf = parser.parse_args()
img_arr_path = argsdeouf.img_arr_path
job_dir = argsdeouf.job_dir


################### load data ##################################### 

X_train = []


img_ar_file = file_io.FileIO(img_arr_path, mode='rb')

X_traing = pickle.load(img_ar_file, encoding='bytes')


##############################   MODEL   ################################
#### def discriminator

disc_model = Sequential()
disc_model.add(Conv2D(32, kernel_size=(5, 5),padding='same', input_shape=(424,424,3)))
disc_model.add(LeakyReLU(0.2))
disc_model.add(Conv2D(64, kernel_size=(5, 5),padding='same'))
disc_model.add(LeakyReLU(0.2))
disc_model.add(Dropout(0.3))
disc_model.add(Conv2D(128, kernel_size=(5, 5), padding='same'))
disc_model.add(LeakyReLU(0.2))
disc_model.add(Dropout(0.3))
disc_model.add(Flatten())
disc_model.add(Dense(1, activation='sigmoid'))

disc_model.compile(loss='binary_crossentropy', optimizer='adam')
disc_model.summary()


##### def generator

generator_model = Sequential()
generator_model.add(Dense(128*106*106, input_dim=100))
generator_model.add(LeakyReLU(0.2))
generator_model.add(Reshape((106, 106, 128)))
generator_model.add(UpSampling2D(size=(2, 2)))
generator_model.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
generator_model.add(LeakyReLU(0.2))
generator_model.add(UpSampling2D(size=(2, 2)))
generator_model.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))

generator_model.compile(loss='binary_crossentropy', optimizer='adam')
generator_model.summary()


### def adversarial model


disc_model.trainable = False
adv_input = Input(shape=(100,))
x = generator_model(adv_input)
adv_output = disc_model(x)
adv_model = Model(inputs=adv_input,outputs=adv_output)
adv_model.compile(loss='binary_crossentropy', optimizer='adam')
adv_model.summary()


################################    train   ##################################
print("On commence l'entrainement")
A_loss = []
for i in range(5000):
	print ("nb -> " + str(i))
	noise = np.random.uniform(-1, 1, size=[256,100])
	image_fake = generator_model.predict(noise)
	imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=256)]
	
	y_true = np.ones(256)
	y_true *= 0.9
	y_false = np.zeros(256)

	x = np.concatenate([imageBatch, image_fake])
	y = np.concatenate([y_true, y_false])
	
	disc_model.trainable = True
	d_loss = disc_model.train_on_batch(x,y)
	
	noise = np.random.uniform(-1, 1, size=[256, 100])
	y = np.ones(256)
	
	disc_model.trainable = False
	a_loss = adv_model.train_on_batch(noise, y)
	A_loss.append(a_loss)

print("loss : "+str(A_loss))
noise = np.random.uniform(-1, 1, size=[256, 100])
print("noise : " +str(noise))
generated_images = generator_model.predict(noise)

pklname = "images.pkl"
pickle.dump(generated_images, open(pklname, "wb"), protocol=4)
with file_io.FileIO(pklname, mode='rb') as input_file:
	with file_io.FileIO(job_dir + "/" +pklname, mode='w+') as output_file:
		output_file.write(input_file.read())

i=0
for image in generated_images:
	image =image.reshape((424,424),3)
	img = image.astype("float32")*255
	img = img.astype("uint8")
	img = PIL.Image.fromarray(img,"RGB")
	imname = "genim_"+str(i)+".png"
	img.save(imname, "PNG")
	with file_io.FileIO(imname, mode='rb') as input_file:
		with file_io.FileIO(job_dir + "/" +imname, mode='w+') as output_file:
			output_file.write(input_file.read())
	i+=1

generator_model.save("genmod.h5")
with file_io.FileIO("genmod.h5", mode='rb') as input_file:
		with file_io.FileIO(job_dir + "/" + "genmod.h5", mode='w+') as output_file:
			output_file.write(input_file.read())

