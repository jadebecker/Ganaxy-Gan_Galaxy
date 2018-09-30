from glob import glob
from PIL import Image
import numpy as np
import pickle
import argparse
from tensorflow.python.lib.io import file_io

parser = argparse.ArgumentParser()
parser.add_argument( '--job-dir')
argsdeouf = parser.parse_args()
job_dir = argsdeouf.job_dir


imgs = glob("images/*")

X_train = []

for img in imgs:
	image = Image.open(img).convert("RGB")
	image = np.array(image)
	image = image.astype('float32')
	image = image / 255
	X_train.append(image)
X_train = np.array(X_train)
#print(X_train.shape)


pklname = "galaxy.pkl"
pickle.dump(X_train, open(pklname,"wb"), protocol =4)
with file_io.FileIO(pklname, mode='rb') as input_file:
	with file_io.FileIO(job_dir + "/" +pklname, mode='w+') as output_file:
		output_file.write(input_file.read())