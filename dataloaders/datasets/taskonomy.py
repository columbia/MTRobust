import torch.utils.data as data

from PIL import Image
import os
import os.path
import zipfile as zf
import io
import logging
import random
import copy
import numpy as np
import time
import torch
import sys
from dataloaders.utils import decode_segmap
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import multiprocessing

import torchvision.transforms as transforms

#
# from data.task_data_loading import

from dataloaders.new.data.task_data_loading import load_target

# Sample usage in taskgrouping repository: https://github.com/tstandley/taskgrouping/blob/master/train_taskonomy.py
# train_dataset = TaskonomyLoader(
#         args.data_dir,
#         task_set=taskonomy_tasks,
#         model_whitelist='train_models.txt',
#         model_limit=args.model_limit,
#         output_size = (args.image_size,args.image_size),
#         augment=True)

class TaskonomyLoader(data.Dataset):

	NUM_CLASSES = 21 #TODO: Change the number of classes.
	INFO = {"mean": [0.485, 0.456, 0.406],
			"std": [0.229, 0.224, 0.225]} #TODO: Change the info
	def __init__(self,
				 root,
				 is_training,
				 threshold,
				 task_set,
				 model_whitelist=None,
				 model_limit=None,
				 output_size=None,
				 convert_to_tensor=True,
				 return_filename=False,
				 augment=False):
		self.root = root
		self.model_limit = model_limit
		self.records = []
		# if model_whitelist is None:
		#     self.model_whitelist = None
		# else:
		#     self.model_whitelist = set()
		#     with open(model_whitelist) as f:
		#         for line in f:
		#             self.model_whitelist.add(line.strip())
		print('root', root)

		for i, (where, subdirs, files) in enumerate(os.walk(os.path.join(root, 'rgb'))): #there is a folder rgb and images are stored there
			print(where)
			print('len files', len(files))
			print('len subdir', subdirs)
			if subdirs != []: continue
			model = where.split('/')[-1]
			print('model dataset', model)
			for each_file in files:
				name_list = each_file.split('_')
				if is_training and int(name_list[1]) < threshold:
					self.records.append(os.path.join(where, each_file))

				elif not is_training and int(name_list[1]) >= threshold:
					self.records.append(os.path.join(where, each_file))

			# if self.model_whitelist is None or model in self.model_whitelist:
			#     full_paths = [os.path.join(where, f) for f in files] #gets the absolute paths for the image files
			#     if isinstance(model_limit, tuple): #model limit defines the images. if single number => take the single image. if tuple => range.
			#         full_paths.sort()
			#         full_paths = full_paths[model_limit[0]:model_limit[1]]
			#     elif model_limit is not None:
			#         full_paths.sort()
			#         full_paths = full_paths[:model_limit]
			#      += full_paths #the relevant paths are thus stored in self.records

		self.configs = {}

		for task in task_set:
			import importlib; config = importlib.import_module('dataloaders.new.configs.{}.config'.format(task))

			cfg = config.get_cfg()
			self.configs[task] = cfg
			# cleanup
			try:
				del config
			except:
				pass

		self.task_set = task_set
		self.output_size = output_size
		self.convert_to_tensor = convert_to_tensor
		self.return_filename = return_filename
		self.to_tensor = transforms.ToTensor()
		self.augment = augment
		if augment:
			print('Data augmentation is on (flip).')
		self.last = {}

	def process_image(self, im, flip):
		if len(im.shape) == 2:
			im = np.expand_dims(im, axis=-1)

		if len(im.shape) == 3: # If im is actually an image
			im = im.transpose((2, 0, 1))
			if flip: im = im[:, ::-1, :]

		im = torch.from_numpy(im).float()

		return im

	def normalize(self, im):
		# composed_transforms = transforms.Compose([
		#     # tr.RandomHorizontalFlip(),
		#     # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
		#     # tr.RandomGaussianBlur(),
		#     tr.Normalize(mean=self.INFO['mean'], std=self.INFO['std']),
		#     tr.ToTensor()])
		img = im.float()
		# img /= 255.0
		img -= torch.FloatTensor(self.INFO['mean']).unsqueeze(1).unsqueeze(1)
		img /= torch.FloatTensor(self.INFO['std']).unsqueeze(1).unsqueeze(1)
		# print("shape ", img.size(),np.array(self.INFO['mean']).shape,torch.FloatTensor(self.INFO['mean']).unsqueeze(1).unsqueeze(1).size())
		# img /= np.array(self.INFO['std'])
		return img

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (image, target) where target is an uint8 matrix of integers with the same width and height.
		If there is an error loading an image or its labels, simply return the previous example.
		"""
		file_name = self.records[index]
		save_filename = file_name

		flip = (random.randint(0, 1) > .5 and self.augment)

		cfg = next(iter(self.configs.values()))
		im, _, _ = cfg['preprocess_fn'](file_name.replace('rgb', '{domain}'), cfg)

		im = self.process_image(im, flip)

		ys = {}
		masks = {}

		for task in self.task_set:
			# print("\n","Reading label", i,"\n")

			cfg = self.configs[task]

			try:
				yim, mask = load_target(file_name.replace('rgb', '{domain}'), im, cfg)
			except:
				print("------------Error : -----------", cfg)
				raise FileNotFoundError

			yim  = self.process_image(yim, flip)
			if task == 'normal' and flip: yim[0] *= -1.0

			if not isinstance(mask, float):
				mask = self.process_image(mask, flip)

			ys[task] 	= yim
			masks[task] = mask

		ys['rgb'] = im

		# print("\n")
		return im, ys, masks

	def __len__(self):
		return (len(self.records))

def resize():
	"""
	TO RESIZE THE WHOLE DATASET: check again
	:return:
	"""
	root_deep = "/mnt/md0/2019Fall/taskonomy/taskonomy-sample-model-1-master"
	root_amogh = "/home/amogh/data/taskonomy-sample-model-1"
	root = "/home/ubuntu/taskonomy-sample-model-1"
	taskonomy_train = TaskonomyLoader(root=root_amogh,
									  is_training=False,
									  threshold=1200,
									  task_set=["depth_zbuffer", "edge_texture", "keypoints2d", "normal", "reshading",
												"keypoints3d", "depth_euclidean", "edge_occlusion",
												"principal_curvature"],
									  # task_set=["segmentsemantic", "depth_zbuffer", "edge_texture", "keypoints2d",
										# 		"normal", "reshading"],
									  model_whitelist=None,
									  model_limit=30,
									  output_size=None)

	dataloader = torch.utils.data.DataLoader(taskonomy_train, batch_size=1, shuffle=False, num_workers=1)
	pil_tr = transforms.ToPILImage()
	for ii, sample in enumerate(dataloader):

		save_resized_images = True
		if save_resized_images:
			im_path = sample[3][0]
			dict_targets = sample[1]
			for type, im in dict_targets.items():
				if type in ["depth_zbuffer", "edge_texture", "keypoints2d", "normal", "reshading", "keypoints3d",
							"depth_euclidean", "edge_occlusion", "principle_curvature"]:
					path_target_output_orig = im_path.replace('rgb', type)
					path_target_output = path_target_output_orig.replace('taskonomy-sample-model-1',
																		 'taskonomy-sample-model-1-small')
					output_im = pil_tr(im[0])
					output_dir = os.path.dirname(path_target_output)
					# if not os.path.exists(output_dir):
					# 	os.makedirs(output_dir)

					# output_im.save(path_target_output)

					# COMPARING THE TWO KINDS OF LABELS AND THE STATISTICS
					original_label = Image.open(path_target_output_orig)
					new_label = output_im
					original_label_array = np.array(original_label)
					new_label_array = np.array(new_label)
					print("For task ", type)
					print(
						"Original label size: ", original_label.size, "\n",
						"Original label mode: ", original_label.mode, "\n",
						"Original label min, mean, max: ", original_label_array.min(), original_label_array.mean(),
						original_label_array.max(), "\n",
						"Tensor mode: ", im[0].type(), "\n",
						"Tensor min, mean, max: ", im[0].min(), im[0].mean(), im[0].max(), "\n",
						"New label size: ", new_label.size, "\n",
						"New label mode: ", new_label.mode, "\n",
						"New label min, mean, max: ", new_label_array.min(), new_label_array.mean(),
						new_label_array.max(),
						"\n",
					)
			print("______________________________________________\n")
			continue

def print_tensor_stats():
	root_deep = "/mnt/md0/2019Fall/taskonomy/taskonomy-sample-model-1-master"
	root_amogh = "/home/amogh/data/taskonomy-sample-model-1"
	#root = "/home/ubuntu/taskonomy-sample-model-1-small-master"
	root = "/home/ubuntu/taskonomy-sample-model-1-master"
	taskonomy_train = TaskonomyLoader(root=root,
									  is_training=False,
									  threshold=1200,
									  # task_set=["segmentsemantic", "depth_zbuffer", "edge_texture", "keypoints2d", "normal", "reshading"],
									  # task_set=["segmentsemantic", "depth_zbuffer", "edge_texture", "keypoints2d", "normal", "reshading"],
									  task_set=["depth_zbuffer", "edge_texture", "keypoints2d", "normal", "reshading",
												"keypoints3d", "depth_euclidean", "edge_occlusion",
												"principal_curvature"],
									  model_whitelist=None,
									  model_limit=30,
									  output_size=None)

	dataloader = torch.utils.data.DataLoader(taskonomy_train, batch_size=1, shuffle=False, num_workers=1)
	# pil_tr = transforms.ToPILImage()
	for ii, sample in enumerate(dataloader):

		# CHECKING tensor stats for each task
		dict_targets = sample[1]
		print("Sample file is: ", sample[3])
		for type, im in dict_targets.items():
			if type in ["depth_zbuffer", "edge_texture", "keypoints2d", "normal", "reshading", "keypoints3d",
						"depth_euclidean", "edge_occlusion", "principal_curvature"]:
				im_tensor = dict_targets[type]
				
				print("FOR ", type,"size is ", im_tensor.size())
				print("Min, Max and mean value are: ", im_tensor.min(), im_tensor.max(), im_tensor.mean(), "\n")

		print("___________________\n")

if __name__ == '__main__':
	# resize()
	# print("_____")
	print_tensor_stats()



	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	args.base_size = 513
	args.crop_size = 513

	root_deep = "/mnt/md0/2019Fall/taskonomy/taskonomy-sample-model-1-master"
	root_amogh = "/home/amogh/data/taskonomy-sample-model-1"
	root = "/home/ubuntu/taskonomy-sample-model-1-master"
	taskonomy_train = TaskonomyLoader(root=root_amogh,
									  is_training=False,
									  threshold=1200,

									  # task_set=["segmentsemantic", "depth_zbuffer", "edge_texture", "keypoints2d", "normal", "reshading"],
									  # task_set=["segmentsemantic", "depth_zbuffer", "edge_texture", "keypoints2d", "normal", "reshading"],
									  # task_set=["depth_zbuffer", "edge_texture", "keypoints2d", "normal", "reshading","keypoints3d", "depth_euclidean", "edge_occlusion", "principal_curvature"],
									  task_set=["segmentsemantic", "depth_zbuffer", "edge_texture", "keypoints2d", "normal", "reshading"],

									  model_whitelist=None,
									  model_limit=30,
									  output_size=None)

	dataloader = torch.utils.data.DataLoader(taskonomy_train, batch_size=1, shuffle=False, num_workers=1)
	# pil_tr = transforms.ToPILImage()
	for ii, sample in enumerate(dataloader):

	# CHECKING tensor stats for each task
		dict_targets = sample[1]
		for type, im in dict_targets.items():
			if type in ["depth_zbuffer", "edge_texture", "keypoints2d", "normal", "reshading","keypoints3d", "depth_euclidean", "edge_occlusion", "principal_curvature"]:
				im_tensor = dict_targets[type]
				print("FOR ",type)
				print("Min, Max and mean value are: ", im_tensor.min(), im_tensor.max(), im_tensor.mean(),"\n")

		print("___________________\n")



	# 	save_resized_images = True
	# 	save=True
	# 	if save_resized_images:
	# 		im_path = sample[3][0]
	# 		dict_targets = sample[1]
	# 		for type, im in dict_targets.items():
	# 			if type in ["depth_zbuffer","edge_texture","keypoints2d","normal","reshading","keypoints3d","depth_euclidean","edge_occlusion","principle_curvature"]:
	# 				path_target_output_orig = im_path.replace('rgb', type)
	# 				path_target_output = path_target_output_orig.replace('taskonomy-sample-model-1', 'taskonomy-sample-model-1-small')
	#
	#
	#
	#
	# 				# SEE THE TYPE IN WHICH IMAGE SHOULD BE SAVED AND CONVERT THE TENSOR TO THAT.
	# 				# if rgb convert directly
	# 				im_arr = im.numpy()
	# 				if im_arr[0].shape[0] == 1:
	# 					print("if",im_arr.shape, im_arr[0].shape)
	# 					output_im = Image.fromarray(im_arr[0][0],mode='I')
	# 				else:
	# 					print("else",im_arr.shape, np.moveaxis(im_arr[0],0,2).shape)
	# 					output_im = Image.fromarray(np.moveaxis(im_arr[0],0,2), mode='RGB')
	# 				# output_im = pil_tr(im[0])
	# 				output_dir = os.path.dirname(path_target_output)
	# 				print(output_im.size)
	#
	# 				if save:
	# 					if not os.path.exists(output_dir):
	# 						os.makedirs(output_dir)
	#
	# 					output_im.save(path_target_output)
	#
	# 				# COMPARING THE TWO KINDS OF LABELS AND THE STATISTICS
	# 				compare=False
	# 				if compare:
	# 					original_label = Image.open(path_target_output_orig)
	# 					new_label = output_im
	# 					original_label_array = np.array(original_label)
	# 					new_label_array = np.array(new_label)
	# 					print("For task ",type)
	# 					print(
	# 						"Original label size: ", original_label.size,"\n",
	# 						"Original label mode: ", original_label.mode,"\n",
	# 						"Original label min, mean, max: ", original_label_array.min(),original_label_array.mean(),original_label_array.max(),"\n",
	# 						"Tensor mode: ", im[0].type(),"\n",
	# 						"Tensor min, mean, max: ", im[0].min(), im[0].mean(),im[0].max(),"\n",
	# 						"New label size: ", new_label.size,"\n",
	# 						"New label mode: ", new_label.mode,"\n",
	# 						"New label min, mean, max: ", new_label_array.min(), new_label_array.mean(),new_label_array.max(),
	# 						 "\n",
	# 						  )
	# 		print("______________________________________________\n")
	# 		continue


	####### VISUALIZE THE labels
		visualize = False
		if visualize:
			print(sample[1]['edge_texture'].size(), sample[1]['keypoints2d'].size())

			print(ii,"\n\n",sample[1]['segmentsemantic'].numpy().shape)

			im_rgb = np.moveaxis(sample[1]['rgb'].numpy().squeeze(),0,2) # as shape is 1,3,x,y
			im_label1 = np.moveaxis(sample[1]['normal'].numpy().squeeze(),0,2) # as shape is 1,3,x,y
			im_label2 = (sample[1]['depth_zbuffer'].numpy().squeeze().squeeze()) # as shape is 1,1,x,y
			im_label3 = (sample[1]['edge_texture'].numpy().squeeze().squeeze()) # as shape is 1,1,x,y
			im_label4 = (sample[1]['keypoints2d'].numpy().squeeze().squeeze()) # as shape is 1,1,x,y
			im_label5 = np.moveaxis(sample[1]['reshading'].numpy().squeeze(),0,2) # as shape is 1,3,x,y


			im_label6 = sample[1]['segmentsemantic'].numpy().squeeze().squeeze()

			print('normal 1', sample[1]['normal'].numpy().shape)
			print('depth_zbuffer', sample[1]['depth_zbuffer'].numpy().shape)
			print('edge_texture', sample[1]['edge_texture'].numpy().shape)
			print('keypoints2d', sample[1]['keypoints2d'].numpy().shape)
			print('reshading', sample[1]['reshading'].numpy().shape)
			print('segmentsemantic', sample[1]['segmentsemantic'].numpy().shape)
			print('seg 2', im_label2.shape)
			import matplotlib.pyplot as plt
			f, axarr = plt.subplots(2, 4)

			print('sample', sample[1])



			axarr[0, 0].imshow(im_rgb)
			axarr[0, 0].set_title("rgb")
			axarr[0, 1].imshow(im_label1)
			axarr[0, 1].set_title("normal")
			axarr[0, 2].imshow(im_label2, cmap='gray')
			axarr[0, 2].set_title("depth_zbuffer")
			axarr[1, 0].imshow(im_label3, cmap='gray')
			axarr[1, 0].set_title("edge_texture")
			axarr[1, 1].imshow(im_label4, cmap='gray')
			axarr[1, 1].set_title("keypoints2d")
			axarr[1, 2].imshow(im_label5)
			axarr[1, 2].set_title("reshading")
			axarr[1, 3].imshow(im_label6)
			axarr[1, 3].set_title("segment")
			# axarr[2].imshow(im_label2, cmap='gray')
			# axarr[3].imshow(vis_adv_img_tr)
			# # print()
			# axarr[4].imshow(diff_img1)
			# axarr[5].imshow(diff_img2)
			plt.show()

			# image = plt.imread("/home/amogh/data/taskonomy-sample-model-1/normal/point_1004_view_0_domain_normal.png")
			# print("\n\n\n____",image,"\n\n\n____")
			# train_dataset = TaskonomyLoader(
			#     args.data_dir,
			#     task_set=taskonomy_tasks,
			#     model_whitelist='train_models.txt',
			#     model_limit=args.model_limit,
			#     output_size=(args.image_size, args.image_size),
			#     augment=True)
			# for jj in range(sample["image"].size()[0]):
			#     img = sample['image'].numpy()
			#     gt = sample['label'].numpy()
			#     tmp = np.array(gt[jj]).astype(np.uint8)
			#     segmap = decode_segmap(tmp, dataset='pascal')
			#     img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
			#     img_tmp *= (0.229, 0.224, 0.225)
			#     img_tmp += (0.485, 0.456, 0.406)
			#     img_tmp *= 255.0
			#     img_tmp = img_tmp.astype(np.uint8)
			#     plt.figure()
			#     plt.title('display')
			#     plt.subplot(211)
			#     plt.imshow(img_tmp)
			#     plt.subplot(212)
			#     plt.imshow(segmap)

			# if ii == 1:
			#     break