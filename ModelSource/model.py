import tensorflow as tf 
import numpy as np 
import argparse
import cv2
import time
from network import Network

#preprocess image
def preprocess(img):
	imgpre = np.copy(img)
	#bgr to rgb
	imgpre = imgpre[..., ::-1]
	#shape (h, w, d) to  (1, h, w, d)
	imgpre = imgpre[np.newaxis,:,:,:]
	imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

	return imgpre


def readImage(content, style):

	#bgr image
	def read(img):
		image = cv2.imread(img, cv2.IMREAD_COLOR)
		image = image.astype(np.float32)
		return image

	srcimg = read(content)
	styleimg = read(style)

	return srcimg, styleimg

def get_content_image(img, parser):

	h,w,d = img.shape
	mx = parser.max_size

	if h > w and h > mx:
		w = (float(mx) / float(h)) * w
		img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
	if w > mx:
		h = (float(mx) / float(w)) * h
		img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)

	img = preprocess(img)

	return img

def get_style_image(content_img, styleimg):

	_, h, w, d = content_img.shape
	#resize the style image
	img = cv2.resize(styleimg, dsize=(h, w), interpolation=cv2.INTER_AREA)
	img = preprocess(img)

	return img


def applyStyle(content_image, style_image, parser, architecture):
	#start the tensorflow Session
	with tf.Session() as sess:
		#build the architecture
		network = architecture.buildModel(content_image)

def main():
	desc = "Tensorflow Implementation for Neural Style Transfer"
	parser = argparse.ArgumentParser(description=desc)

	#options
	parser.add_argument('--verbose', action='store_true',
		help='Boolean flag indicating that statement should be printing in the console')

	#root path
	parser.add_argument('--rootdir', type=str,
		default = 'F:/work projects/next19docs/app/',
		help='Root Directory Path (Image Content)')

	parser.add_argument('--baseimg', type=str,
		default = 'images/content_image.jpg',
		help= 'Path location for Content Image')

	parser.add_argument('--styleimg', type=str,
		default= 'images/style_image.jpg',
		help='Path location for Style Image')

	parser.add_argument('--max_size', type=int,
		default=512,
		help='Maximum width or Height of the image')


	#add model weights
	parser.add_argument('--modelweights', type=str,
		default='imagenet-vgg-verydeep-19.mat',
		help='Weights and Biases of the VGG-19 Network')


	#pool layers
	parser.add_argument('--poolingChoice', type=str,
		default='max',
		help='Choose between MaxPooling or Average_pooling')


	args = parser.parse_args()
	architecture = Network(args) #creating an object
	#get the parser inputs
	contentimagePath = args.rootdir + args.baseimg
	styleimagePath = args.rootdir + args.styleimg

	srcImg, styleImg =readImage(contentimagePath, styleimagePath)

	#get content image
	content_image = get_content_image(srcImg, args) #(1, 512, 288, 3)
	style_image = get_style_image(content_image, styleImg) #(1, 288, 512, 3)

	with tf.Graph().as_default():
		print("\n ------RENDERIN SINGLE IMAGE--------- \n")
		tick = time.time()
		applyStyle(content_image, style_image, args, architecture)
		tock = time.time()
		print("Single Image Elapsed Tune:{}".format(tock - tick))


#main function calling
if __name__ == '__main__':
	main()