import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import caffe
from sklearn.externals import joblib
import argparse


class DetectorSigNet(caffe.Net):

	def __init__(self, model_file, pretrained_file, channel_swap=[2,1,0]):
		caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

		# configure pre-processing
		in_ = self.inputs[0]
		self.transformer = caffe.io.Transformer({in_: self.blobs[in_].data.shape})
		self.transformer.set_transpose(in_, (2, 0, 1))
		if channel_swap is not None:
			self.transformer.set_channel_swap(in_, channel_swap)

		self.minSize=16
		self.maxSize=200
		self.maxProportion=1.2

	def get_mser_windows_list(self, img):
		image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		padding = 5
		mser = cv2.MSER(_delta=1)
		mser_areas = mser.detect(image)

		windows_list = []
		for blob in mser_areas:
			df_blob = pd.DataFrame(blob)
			xmin, ymin = df_blob.min(0)
			xmax, ymax = df_blob.max(0)

			xmin -= padding
			ymin -= padding
			xmax += padding
			ymax += padding
			
			if xmin >= 0 and ymin >= 0 and xmax <= img.shape[1] and ymax <= img.shape[0]:
				windows_list.append((xmin, ymin, xmax, ymax))

		return windows_list

	def get_valid_windows_list(self, image):
		valid_windows = []

		windows = self.get_mser_windows_list(image)
		for window in windows:
			size_x = float(window[2] - window[0])
			size_y = float(window[3] - window[1])
			if size_x < self.minSize or size_y < self.minSize or size_x > self.maxSize or size_y > self.maxSize or\
			size_x / size_y > self.maxProportion or size_y / size_x > self.maxProportion:
				continue
			valid_windows.append(window)

		return valid_windows

	def detect_windows(self, image_fname):
		window_valids = []
		window_inputs = []

		image = plt.imread(image_fname)
		for window in self.get_valid_windows_list(image):
			window_valids.append(window)
			window_inputs.append(image[window[1]:window[3], window[0]:window[2]])

		in_ = self.inputs[0]
		caffe_in = np.zeros((len(window_inputs), window_inputs[0].shape[2])+ self.blobs[in_].data.shape[2:], dtype=np.float32)
		for ix, window_in in enumerate(window_inputs):
			caffe_in[ix] = self.transformer.preprocess(in_, window_in)
		out = self.forward_all(**{in_: caffe_in})
		predictions = out[self.outputs[0]].squeeze(axis=(2, 3))
		# predictions = out[self.outputs[0]].squeeze()

		detections = []
		ix = 0
		for window in window_valids:
			detections.append({'window': window, 'prediction': predictions[ix], 'filename': image_fname})
			ix += 1
		return detections


class FeaturesSigNet(caffe.Net):

	def __init__(self, model_file, pretrained_file, channel_swap=[2,1,0]):
		caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

		# configure pre-processing
		in_ = self.inputs[0]
		self.transformer = caffe.io.Transformer({in_: self.blobs[in_].data.shape})
		self.transformer.set_transpose(in_, (2, 0, 1))
		if channel_swap is not None:
			self.transformer.set_channel_swap(in_, channel_swap)

	def get_features(self, image_fname, windows):
		image = plt.imread(image_fname)
		window_inputs = []
		for window in windows:
			window_inputs.append(image[window[1]:window[3], window[0]:window[2]])

		in_ = self.inputs[0]
		caffe_in = np.zeros((len(window_inputs), window_inputs[0].shape[2])+ self.blobs[in_].data.shape[2:], dtype=np.float32)
		for ix, window_in in enumerate(window_inputs):
			caffe_in[ix] = self.transformer.preprocess(in_, window_in)
		out = self.forward_all(**{in_: caffe_in})
		# predictions = out[self.outputs[0]].squeeze()
		predictions = out[self.outputs[0]]

		features = []
		ix = 0
		for ix in range(len(window_inputs)):
			features.append(predictions[ix].flatten())
			ix += 1
		return np.array(features)	


def bootstrap(windows):
	if len(windows) == 1:
		return windows

	epsilon = 20

	groups = {}
	indexes = range(len(windows))
	i = 0
	while i < len(windows):
		groups[i] = [i]
		indexes.remove(i)
		to_remove = []
		for j in indexes:
			if (np.abs(np.array(windows[i]) - np.array(windows[j])) < epsilon).all():
				groups[i].append(j)
				to_remove.append(j)
		for k in to_remove:
			indexes.remove(k)
		if indexes:
			i = indexes[0]
		else:
			break

	res_windows = []
	for key in groups:
		xmin = np.min([windows[ind][0] for ind in groups[key]])
		ymin = np.min([windows[ind][1] for ind in groups[key]])
		xmax = np.max([windows[ind][2] for ind in groups[key]])
		ymax = np.max([windows[ind][3] for ind in groups[key]])
		res_windows.append((xmin, ymin, xmax, ymax))

	return res_windows


def detect_traffic_signs(model_def, model_features_def, pretrained_model, threshold=0.55, image_fname=None, images_folder=None, gt_file=None, gpu=True):
	if image_fname == None and images_folder == None:
		print 'Give image_fname for detection on particular image or images_folder for detection on random image in folder'
		return

	if not image_fname and images_folder:
		img_ind = np.random.choice(900)
		if img_ind < 10:
			img_file = '0000%d.ppm' % img_ind
		elif img_ind < 100:
			img_file = '000%d.ppm' % img_ind
		else:
			img_file = '00%d.ppm' % img_ind

		image_fname = images_folder + '/' + img_file

	if gpu:
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()
	detector = DetectorSigNet(model_def, pretrained_model)
	prediction = detector.detect_windows(image_fname)

	windows = []
	sign_ind = []
	for ix in range(len(prediction)):
		pred_prob = prediction[ix]['prediction']
		if pred_prob[1] > threshold:
			windows.append(prediction[ix]['window'])

	if windows:
		features_detector = FeaturesSigNet(model_features_def, pretrained_model)
		X = features_detector.get_features(image_fname, windows)
		clf = joblib.load('classifier/clf2/classifier.pkl')
		y = clf.predict(X)

		for i in range(y.shape[0]):
			if y[i] == 1:
				sign_ind.append(i)

	sign_windows = []
	for ind in sign_ind:
		sign_windows.append(windows[ind])
	frames = bootstrap(sign_windows)

	img = plt.imread(image_fname)
	plt.imshow(img)
	currentAxis = plt.gca()
	for frame in frames:
		coords = (frame[0], frame[1]), frame[2] - frame[0], frame[3] - frame[1]
		currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='r', linewidth=3))
	
	if gt_file:
		short_fname = os.path.split(image_fname)[1]
		gt_list = []
		with open(gt_file) as f:
			for line in f:
				words = line.strip().split(';')
				if words[0] == short_fname:
					gt_list.append(np.array(words[1:5], dtype='float'))
		for key in range(len(gt_list)):
			gt = gt_list[key]
			coords = (gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1]
			currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='g', linewidth=3))

	plt.show()


def main(args):
	parser = argparse.ArgumentParser()

	parser.add_argument("--model_def", 
		default='model/signet_nin_finetune_deploy.prototxt',
		help="Model definition file.")
	parser.add_argument("--model_features_def", 
		default='model/signet_nin_finetune_features.prototxt',
		help="Model of feature detector definition file.")
	parser.add_argument("--pretrained_model",
		default='model/caffemodel/signet_nin_finetune3.caffemodel',
		help="Trained model weights file.")
	parser.add_argument("--threshold",
		type=float,
		default=0.55,
		help="Treshhold for CNN.")
	parser.add_argument("--image_fname",
		default=None,
		help="Image file name for traffic signs detection.")
	parser.add_argument("--images_folder",
		default=None,
		help="Image files folder for traffic signs detection on random image from this folder.")
	parser.add_argument("--gt_file",
		default=None,
		help="txt file with ground truth regions for plot them. Format of file similar with one from GTSDB competition")
	parser.add_argument("--gpu",
		action='store_true',
		help='Use GPU for computation')

	args = parser.parse_args()

	if args.gpu:
		gpu = True
	else:
		gpu = False


	detect_traffic_signs(model_def=args.model_def, model_features_def=args.model_features_def, pretrained_model=args.pretrained_model,
	 threshold=args.threshold, image_fname=args.image_fname, images_folder=args.images_folder, gt_file=args.gt_file, gpu=gpu)


if __name__ == '__main__':
	import sys
	main(sys.argv)


