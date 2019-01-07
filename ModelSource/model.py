import tensorflow as tf 
import numpy as np 
import argparse
import cv2
import time
import os

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


#<!---postprocess-->
def postprocess(img):

	imgpost = np.copy(img)
	imgpost += np.array([123.68, 116.779, 103.939]).reshape(1,1,1,3)

	#shape (1, h, w, d) to (h, w, d)
	imgpost = imgpost[0]
	imgpost = np.clip(imgpost, 0, 255).astype('uint8')

	#rbg to bgr
	imgpost = imgpost[...,::-1]

	return imgpost



#<!--get noise image--->
def get_noise_image(ratio, content_img, parser):
	#get random seed
	np.random.seed(parser.seed)
	noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
	img = ratio * noise_img + (1. - ratio) * content_img
	return img


#<!--get Initial Image-->
def get_init_image(init_type, content_img, style_img, parser, frame=None):
	#condition
	if  init_type == 'content':
		return content_img
	elif init_type == 'style_img':
		return style_img
	elif init_type == 'random':
		init_img = get_noise_image(parser.noise_ratio, content_img, parser)
		return init_img


def readImage(content, style):

	#bgr image
	def read(img):
		image = cv2.imread(img, cv2.IMREAD_COLOR)
		picture = image.astype(np.float32)
		return picture

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

def get_style_image(content_img, styleImage):

	_, h, w, d = content_img.shape
	style_imgs = []

	#resize the style image
	img = cv2.resize(styleImage, dsize=(w, h), interpolation=cv2.INTER_AREA)
	img = preprocess(img)
	style_imgs.append(img)

	return style_imgs

#<!--Gram Matrix-->
def gram_matrix(x, area, depth):
	F = tf.reshape(x, (area, depth))
	G = tf.matmul(tf.transpose(F), F)

	return G


#<!---Style Layer Loss-->
def style_layer_loss(a, x):
	_, h, w, d = a.get_shape()
	M = h.value * w.value
	N = d.value
	A = gram_matrix(a, M, N)
	G = gram_matrix(x, M, N)

	loss = (1. / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G-A), 2))

	return loss


#<!--content layer loss-->
def content_layer_loss(p, x, parser):
	_, h, w, d = p.get_shape()
	M = h.value * w.value
	N = d.value
	if parser.content_loss_function == 1:
		K = 1. / (2. * N**0.5 * M**0.5)
	elif parser.content_loss_function == 2:
		K = 1. / (N * M)
	elif parser.content_loss_function == 3:
		K = 1. / 2.
	
	loss = K * tf.reduce_sum(tf.pow((x - p), 2))

	return loss

	
#<!--Loss Layer for normal content image-->
def sum_style_losses(sess, net, style_imgs, parser):
	  total_style_loss = 0.
	  weights = parser.style_imgs_weights

	  #iteration starts
	  for img, img_weight in zip(style_imgs, weights):
	    sess.run(net['input'].assign(img))
	    style_loss = 0.

	    #iterate through layers and weights
	    for layer, weight in zip(parser.style_layers, parser.style_layer_weights):
	      a = sess.run(net[layer])
	      x = net[layer]
	      a = tf.convert_to_tensor(a)
	      style_loss += style_layer_loss(a, x) * weight

	    style_loss /= float(len(parser.style_layers))
	    total_style_loss += (style_loss * img_weight)

	  total_style_loss /= float(len(style_imgs))
	  return total_style_loss


#<!--Sum content losses-->
def sum_content_losses(sess, net, content_img, parser):
	#run the session
	sess.run(net['input'].assign(content_img))
	content_loss = 0.
	#iterate through the content layers
	for layer, weight in zip(parser.content_layers, parser.content_layer_weights):
		p =  sess.run(net[layer])
		x =  net[layer]
		p =  tf.convert_to_tensor(p)
		content_loss += content_layer_loss(p, x, parser) * weight

	content_loss /= float(len(parser.content_layers))
	return content_loss


#<!--Loss layer for Masked Images-->
def sum_masked_style_losses(sess, net, style_imgs, parser):

	#<!---Custom Functions for Mask Style Layer-->
	def mask_style_layer(a, x, mask_img):
		_, h, w, d = a.get_shape()
		mask = get_mask_image(mask_img, w.value, h.value)
		mask = tf.convert_to_tensor(mask)
		tensors = []

		for _ in range(d.value):
			tensors.append(mask)

		mask = tf.stack(tensors, axis=2)
		mask = tf.stack(mask, axis=0)
		a = tf.multiply(a, mask)
		x = tf.multiply(x, mask)

		return a, x


	total_style_loss = 0.
	weights = parser.style_imgs_weights
	masks = parser.style_mask_imgs

	for img, img_weight, img_mask in zip(style_imgs, weights, masks):

		#run the network architecture
		sess.run(net['input'].assign(img))
		style_loss = 0.

		for layer, weight in zip(parser.style_layers, parser.style_layer_weights):
			a = sess.run(net[layer])
			x = net[layer]
			a = tf.convert_to_tensor(a)
			a, x = mask_style_layer(a, x, img_mask)
			style_loss += style_layer_loss(a, x) * weight

		style_loss /= float(len(parser.style_layers))
		total_style_loss += (style_loss * img_weight)

	total_style_loss /= float(len(style_imgs))

	return total_style_loss


#<!---Get Optimizer-->
def get_optimizer(loss, parser):

	#printing the iterations
	print_iterations = args.print_iterations if True else 0

	if parser.optimizer == 'lbfgs':
		optimizer = tf.contrib.opt.ScipyOptimizerInterface(
			loss, method='L-BFGS-B',
			options={'maxiter': args.max_iterations,
					'disp': print_iterations})

	elif parser.optimizer == 'adam':
		 optimizer = tf.train.AdamOptimizer(parser.learning_rate)

	return optimizer



#<!--Minimize with Adam--->
def minimize_with_adam(sess, net, optimizer, init_img, loss, parser):

	print("\n ---- MINIMIZING LOSS FUNCTION USING: ADAM OPTIMIZER---- \n")
	train_op = optimizer.minimize(loss)
	init_op = tf.global_variables_initializer()
	sess.run(init_op) #initialize the operations
	sess.run(net['input'].assign(init_img))
	iterations = 0

	while(iterations < parser.max_iterations):
		sess.run(init_op)

		if iterations % args.print_iterations == 0 and parser.verbose:
			curr_loss = loss.eval()
			print("At Iteration:{} \t loss= {}".format(iterations, curr_loss))

		iterations += 1 #increment

#<!--Minimize with LBFGS--->
def minimize_with_lbfgs(sess, net, optimizer, init_img):
	
	print("\n ---- MINIMIZING LOSS FUNCTION USING : L-BFGS OPTIMIZER ----\n")
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	sess.run(net['input'].assign(init_img))
	optimizer.minimize(sess)


#<!---convert to original colors-->
def convert_to_original_colors(content_img, stylized_img, parser):
	content_img = postprocess(content_img)
	stylized_img = postprocess(stylized_img)

	if parser.color_convert_type == 'yuv':
		cvt_type = cv2.COLOR_BGR2YUV
		inv_cvt_type = cv2.COLOR_YUV2BGR

	elif parser.color_convert_type == 'ycrcb':
		cvt_type = cv2.COLOR_BGR2YCR_CB
		inv_cvt_type = cv2.COLOR_YCR_CB2BGR

	elif parser.color_convert_type == 'luv':
		cvt_type = cv2.COLOR_BGR2LUV
		inv_cvt_type = cv2.COLOR_LUV2BGR

	elif parser.color_convert_type == 'lab':
		cvt_type = cv2.COLOR_BGR2LAB
		inv_cvt_type = cv2.COLOR_LAB2BGR

	content_cvt = cv2.cvtColor(content_img, cvt_type)
	stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
	c1, _, _ = cv2.split(stylized_cvt)
	_, c2, c3 = cv2.split(content_cvt)

	merged = cv2.merge((c1, c2, c3))
	dest = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
	dest = preprocess(dest)
	return dest


#<!--Write and Image-->
def write_image(path, img):
	img = postprocess(img)
	cv2.imwrite(path, img)


#<!--- write the Image output-->
def write_image_output(output_img, content_image, style_imgs, init_img, parser):

	out_dir = os.path.join(parser.img_output_dir, parser.result_name)
	img_path = os.path.join(out_dir, parser.result_name + '.png')
	content_path = os.path.join(out_dir, 'content.png')
	init_path = os.path.join(out_dir, 'init.png')

	#write the image
	write_image(img_path, output_img)
	write_image(content_path, content_image)
	write_image(init_path, init_img)

	#save the configuration settings
	out_file  = os.path.join(out_dir, 'meta_data.txt')
	f = open(out_file, 'w') #open the file in write mode
	f.write("Image_Name:{}\n".format(parser.result_name))
	f.write("Content Image Name:{}\n".format(parser.content_img))
	f.write("Init Type Name:{}\n".format(parser.init_img_type))
	f.write("Content Weight:{}\n".format(parser.content_weight))
	f.write("Style Weight:{}\n".format(parser.style_weight))
	f.write("tv_weight:{}\n".format(parser.tv_weight))
	f.write("Content Layers:{}\n".format(parser.content_layers))
	f.write("Stlye Layers:{}\n".format(parser.style_layers))
	f.write("Optimizer Type:{}\n".format(parser.optimizer))
	f.write("Max Iterations:{}\n".format(parser.max_iterations))
	f.write("Max Image Size:{}\n".format(parser.max_size))

	f.close()


#<!-- Custom Function for Applying Style--->
def applyStyle(content_image, style_image, parser, architecture, init_img):
	#start the tensorflow Session
	#run every operations within the session
	with tf.Session() as sess:
		#build the architecture
		network = architecture.buildModel(content_image)
		print("\n---- MODEL ARCHITECTURE IS DEFINED AND SUCCESFULLY LOADED----\n")
		#style loss
		if parser.style_mask:
			L_style = sum_masked_style_losses(sess, network, style_image, parser)
		else:
			L_style = sum_style_losses(sess, network, style_image,  parser)

		#<!-- content loss-->
		L_content = sum_content_losses(sess, network, content_image, parser)

		#<!-- denoising loss-->
		L_tv = tf.image.total_variation(net['input'])

		#<!-- loss weights-->
		alpha = parser.content_weight
		beta = parser.style_weight
		theta = parser.tv_weight

		#total loss
		L_total = alpha * L_content
		L_total += beta * L_style
		L_total += theta * L_tv

		#<!--Optimizer-->
		optimizer = get_optimizer(L_total, parser)

		if parser.optimizer == 'adam':
			minimize_with_adam(sess, net, optimizer, init_img, L_total, parser)
		elif parser.optimizer == 'lbfgs':
			minimize_with_lbfgs(sess, net, optimizer, init_img)


	output_img = sess.run(net['input'])

	if parser.original_colors:
		output_img = convert_to_original_colors(np.copy(content_image), output_img, parser)

	#<!--write imgage output-->
	write_image_output(output_img, content_image, style_image, init_img, parser)


#<!---Normalization-->
def normalize(weights):
	denom = sum(weights)
	if denom > 0.:
		return [float(i) / denom for i in weights]
	else:
		return [0.] * len(weights)


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

	#get initial image type
	parser.add_argument('--init_img_type', type=str,
		default='content',
		choices=['random', 'content', 'style'],
		help='Image used to Initialize the Network')


	#add model weights
	parser.add_argument('--modelweights', type=str,
		default='../imagenet-vgg-verydeep-19.mat',
		help='Weights and Biases of the VGG-19 Network')


	#pool layers
	parser.add_argument('--poolingChoice', type=str,
		default='max',
		help='Choose between MaxPooling or Average_pooling')

	#style image weights
	parser.add_argument('--style_imgs_weights', type=str,
		default=[1.0],
		help='Interpolation weights of each of the Style Images')

	#content weight
	parser.add_argument('--content_weight', type=str,
		default=5e0,
		help='Maximum for the content loss function')

	#style weight
	parser.add_argument('--style_weight', type=float,
		default=1e-4,
		help='Weight for the temporal loss function')

	#total variational loss
	parser.add_argument('--tv_weight', type=float,
		default=1e-3,
		help='Weight for Total Variation Loss function')

	#temporal weight function
	parser.add_argument('--temporal_weight', type=float,
		default=2e2,
		help='Weight for Temporal loss function')

	#content  loss function
	parser.add_argument('--content_loss_function', type=int,
		default=1,
		choices=[1,2,3],
		help = 'Different Constants for the content layer loss function')

	#content layer
	parser.add_argument('--content_layers', nargs='+', type=str,
		default=['conv4_2'],
		help='VGG19 layers used for the content image')

	#content layer weights
	parser.add_argument('--content_layer_weights', nargs='+', type=str,
		default=[1.0],
		help='Contributions of each layer to loss')


	#style mask images
	parser.add_argument('--style_mask', action='store_true',
		help='Transfer the style to masked regions')

	parser.add_argument('--style_mask_imgs', nargs='+', type=str,
		default=None,
		help='Filenames of the Style Mask Images')


	#style layers
	parser.add_argument('--style_layers', nargs='+', type=str,
		default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
		help='VGG19 Layers used for the Style Image')

	parser.add_argument('--style_layer_weights', nargs='+', type=float,
		default=[0.2, 0.2, 0.2, 0.2, 0.2],
		help='Contributions (weights) of each style layer loss function')

	#noise ration
	parser.add_argument('--noise_ratio', type=float,
		default=1.0,
		help='Interpolation value between the content image and noise image if the network is initialized with random weights')

	#seed
	parser.add_argument('--seed', type=int,
		default=0,
		help='Seed for random number generator')


	#learning rate
	parser.add_argument('--learning_rate', type=int,
		default=1e0,
		help='Learning Rate Parameter for the Adam optimizer')

	#max iterations
	parser.add_argument('--max_iterations', type=int,
		default=1000,
		help='Maximum number of iterations for the Optimizer')

	#print iterations
	parser.add_argument('--print_iterations', type=int,
		default=50,
		help='Number of iterations between optimizer print statements')


	#color convert type
	parser.add_argument('--color_convert_type', type=str,
		default='yuv',
		choices=['yuv', 'ycrcb', 'luv', 'lab'],
		help='Color Space for Conversion to Original Colors')

	#output directory
	parser.add_argument('--img_output_dir', type=str,
		default='F:/work projects/next19docs/app/ModelSource/image_output',
		help='Relative or Absolute Directory Path to output Image and data')

	#result image name
	parser.add_argument('--result_name', type=str,
		default='result',
		help='Filename of the Output Image')

	#default content image name
	parser.add_argument('--content_img', type=str,
		default='style_image.jpg',
		help='Filename of the Content Image')

	
	args = parser.parse_args()
	architecture = Network(args) #creating an object

	#normalize the weights
	args.style_layer_weights = normalize(args.style_layer_weights)
	args.content_layer_weights = normalize(args.content_layer_weights)
	args.style_imgs_weights = normalize(args.style_imgs_weights)

	#get the parser inputs
	contentimagePath = args.rootdir + args.baseimg
	styleimagePath = args.rootdir + args.styleimg

	srcImg, styleImg =readImage(contentimagePath, styleimagePath)

	#SourceImage Shape - (3264, 1836, 3)
	#StyleImage Shape - (1000, 1500, 3)

	#get content image
	content_image = get_content_image(srcImg, args) #(1, 512, 288, 3)
	style_image = get_style_image(content_image, styleImg) #(1, 288, 512, 3)


	with tf.Graph().as_default():
		print("\n ------RENDERIN SINGLE IMAGE--------- \n")
		init_img = get_init_image(args.init_img_type, content_image, style_image, args)
		tick = time.time()
		applyStyle(content_image, style_image, args, architecture, init_img)
		tock = time.time()
		print("Single Image Elapsed Tune:{}".format(tock - tick))


#main function calling
if __name__ == '__main__':
	main()
