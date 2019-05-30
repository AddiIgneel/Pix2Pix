import os
import cv2
import glob
import time
import torch
import numpy as np
import torchvision
from PIL import Image
from models import create_model
from torchvision.utils import save_image
from options.test_options import TestOptions
import torchvision.transforms.functional as TF

def load_paths(path):
	
	'''
    	Function to get file paths
    	input params:
    	path : String pointing to direction of directory
    	returns:
    	real_A: images list 
    	real_B: list of label images
    '''

	real_A = glob.glob('../base/*.png')
	real_B = glob.glob('../base/*.jpg')
	assert len(real_A)==len(real_B)
	return real_A, real_B 

def extract_pacthes(A):
    
    '''
    	Function extracts four equal sized patches from image
    	input params:
    	A : Image array
    	returns:
    	patches: list of 4 patches obtained from image
    '''
   
    patches = []
    
    patches.append(A[:128, :128, :]) # top left quadrant
    patches.append(A[:128, 128:, :]) # top right quadrant
    patches.append(A[128:, :128, :]) # bottom left quadrant
    patches.append(A[128:, 128:, :]) # bottom right quadrant

    return patches    

def merge_patches(patches):
    
    '''
    	Function to merge patches into single image
    	input params:
    	patches : Array of shape patches, width, height, channels containing 4 patches of image.
    	returns:
    	img: combined single image of all patches
    '''

    img = np.zeros(shape=(256, 256, 3))
    img[:128, :128, :] = patches[0] # top left quadrant
    img [:128, 128:, :] = patches[1] # right quadrant
    img[128:, :128, :] = patches[2] # top left quadrant
    img[128:, 128:, :] = patches[3]# right quadrant
    return img
            
def read_images(files, labels):

	'''
    	Function to read images as array
    	input params:
    	files : List of path of Images to be read
    	labels : List of path of Label Images to be read
    	returns:
    	input_data : Array of images 
    	labels : Array of label images
    '''

	input_data = []
	labels_data = []
	for i,j in zip(files, labels):

		img = cv2.resize(cv2.imread(i), (256,256))
		img1 = cv2.resize(cv2.imread(j), (256,256))
		
		input_data.append(img)	
		labels_data.append(img1)

	return input_data, labels_data

def run_patch(input_data, labels, netG, output_dir):

	'''
		Function runs forward pass on all images by first extracting patches and merging the network results as single image.
		input params:
			input_data : List of image numpy arrays
			netG : network definition with pretrained weights
			labels : the real images corresponding to input
			output_dir: directory to save images
		returns:
			nothing
	'''

	#loss = []
	
	network_output = os.path.join(output_dir, 'network_output.jpg')
	for i, count in zip(input_data, range(len(input_data))):
		image_output = []
		patches = extract_pacthes(i)
		#cv2.imwrite('input_{}.jpg'.format(count), i)
		for p, counter in zip(patches, range(len(patches))):
			#cv2.imwrite('patch_{}_{}.jpg'.format(count,counter), p)
			input_img = cv2.resize(p, (256,256))
			temp = TF.to_tensor(input_img)
			temp.unsqueeze_(0)

			with torch.no_grad():
				fake_B = netG(temp)
				save_image(fake_B, network_output)
		 		#convert tensor to image
				fake_B = fake_B.view(fake_B.shape[1:])
				output_img = np.moveaxis(fake_B.numpy(), 0, -1)
				output_img = cv2.resize(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), (128, 128), interpolation=cv2.INTER_AREA)
				#cv2.imwrite('output_patch_{}_{}.jpg'.format(count, counter), output_img)
				image_output.append(np.array(output_img))
		merged_img = merge_patches(np.array(image_output))	
		cv2.imwrite(os.path.join(output_dir, 'merged_output_{}.jpg'.format(count)), merged_img)
	# 	loss.append(l2_loss(merged_img, labels[count]))
	# print('mean loss: ', np.mean(loss))

def run_all(input_data, netG, output_dir):
	
	'''
		Function runs forward pass on all images and saves the network output as image
		input params:
			input_data : List of image numpy arrays
			netG : network definition with pretrained weights
			output_dir: dir to save output
		returns:
			nothing
	'''
	
	for i, count in zip(input_data, range(len(input_data))):
		temp = TF.to_tensor(i)
		temp.unsqueeze_(0)
		with torch.no_grad():
			fake_B = netG(temp)
			save_image(fake_B, os.path.join(output_dir, 'output_{}.jpg'.format(count)))

if __name__ == '__main__':

	opt = TestOptions().parse()  # get test options
	# hard-code some parameters for test
	opt.num_threads = 0   # test code only supports num_threads = 1
	opt.batch_size = 1    # test code only supports batch_size = 1
	opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
	opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
	opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
	model = create_model(opt)      # create a model given opt.model and other options
	model.setup(opt)               # regular setup: load and print networks; create schedulers
	
	if opt.eval:
		model.eval()
	
	A, B = load_paths(opt.dataroot) 
	# get the model definition from model file
	model_net = model.netG 
	# read the data and labels
	input_data, labels_data = read_images(A, B)
	output_dir = './results'
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	if opt.type_eval == 'without_patches':
		path_to_write = os.path.join(output_dir, 'without_patches')
		if not os.path.exists(path_to_write):
			os.mkdir(path_to_write)

		print('running through whole data set without making patches. Result will be saved to results/without_patches')
		start = time.time()
		run_all(input_data, model_net, path_to_write)
		end = time.time()
		print('Total time taken {}'.format(end-start))

	else:
		path_to_write = os.path.join(output_dir, 'patches')
		if not os.path.exists(path_to_write):
			os.mkdir(path_to_write)

		print('Patches will be generated for each image. Output will be saved to results/patches folder')
		start = time.time()
		run_patch(input_data, labels_data, model_net, path_to_write)
		end = time.time()
		print('total time taken {}'.format(end-start))