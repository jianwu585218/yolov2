"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import glob
import argparse
import pickle
import cv2
import numpy as np
from src.utils import *
from src.yolo_net import Yolo
import csv
import json

# CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
# 		   'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
# 		   'tvmonitor']
CLASSES = ['person']

def get_args():
	parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
	parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
	parser.add_argument("--conf_threshold", type=float, default=0.6)
	parser.add_argument("--nms_threshold", type=float, default=0.45)
	parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
	#以下是分场景训练的模型（20类别），预训练模型是用voc2007
	# parser.add_argument("--pre_trained_model_path_scene1", type=str, default="trained_bot_models/scene1/whole_model_trained_yolo_bot")
	# parser.add_argument("--pre_trained_model_path_scene2", type=str, default="trained_bot_models/scene2/whole_model_trained_yolo_bot")
	# parser.add_argument("--pre_trained_model_path_scene3", type=str, default="trained_bot_models/scene3/whole_model_trained_yolo_bot")
	# parser.add_argument("--pre_trained_model_path_scene4", type=str, default="trained_bot_models/scene4/whole_model_trained_yolo_bot")

	#以下是分场景训练的模型（1类别），预训练模型是用bot训练过的20类模型
	# parser.add_argument("--pre_trained_model_path_scene1", type=str, default="trained_bot_person_models/scene1/whole_model_trained_yolo_bot")
	# parser.add_argument("--pre_trained_model_path_scene2", type=str, default="trained_bot_person_models/scene2/whole_model_trained_yolo_bot")
	# parser.add_argument("--pre_trained_model_path_scene3", type=str, default="trained_bot_person_models/scene3/whole_model_trained_yolo_bot")
	# parser.add_argument("--pre_trained_model_path_scene4", type=str, default="trained_bot_person_models/scene4/whole_model_trained_yolo_bot")
	# parser.add_argument("--pre_trained_model_path_scene5", type=str, default="trained_bot_person_models/scene5/whole_model_trained_yolo_bot")

	# #以下是分场景训练的模型（1类别），预训练模型是用用voc2007
	# parser.add_argument("--pre_trained_model_path_scene1", type=str, default="trained_bot_person_models2/scene1/whole_model_trained_yolo_bot")
	# parser.add_argument("--pre_trained_model_path_scene2", type=str, default="trained_bot_person_models2/scene2/whole_model_trained_yolo_bot")
	# parser.add_argument("--pre_trained_model_path_scene3", type=str, default="trained_bot_person_models2/scene3/whole_model_trained_yolo_bot")
	# parser.add_argument("--pre_trained_model_path_scene4", type=str, default="trained_bot_person_models2/scene4/whole_model_trained_yolo_bot")
	# parser.add_argument("--pre_trained_model_path_scene5", type=str, default="trained_bot_person_models2/scene5/whole_model_trained_yolo_bot")

	#以下是分场景训练的模型（1类别），预训练模型是用用coco14+17
	parser.add_argument("--pre_trained_model_path_scene1", type=str, default="trained_bot_person_models3/scene1/whole_model_trained_yolo_bot")
	parser.add_argument("--pre_trained_model_path_scene2", type=str, default="trained_bot_person_models3/scene2/whole_model_trained_yolo_bot")
	parser.add_argument("--pre_trained_model_path_scene3", type=str, default="trained_bot_person_models3/scene3/whole_model_trained_yolo_bot")
	parser.add_argument("--pre_trained_model_path_scene4", type=str, default="trained_bot_person_models3/scene4/whole_model_trained_yolo_bot")
	parser.add_argument("--pre_trained_model_path_scene5", type=str, default="trained_bot_person_models3/scene5/whole_model_trained_yolo_bot")
	parser.add_argument("--input", type=str, default="test_images")
	parser.add_argument("--output", type=str, default="test_images")
	parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
	args = parser.parse_args()
	return args


def get_multiscene_result(opt,val1_dir,jsonout_dir):
	os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
	if torch.cuda.is_available():
		model_scene1 = torch.load(opt.pre_trained_model_path_scene1)
		model_scene2 = torch.load(opt.pre_trained_model_path_scene2)
		model_scene3 = torch.load(opt.pre_trained_model_path_scene3)
		model_scene4 = torch.load(opt.pre_trained_model_path_scene4)
		model_scene5 = torch.load(opt.pre_trained_model_path_scene5)

	else:
		model_scene1 = torch.load(opt.pre_trained_model_path_scene1, map_location=lambda storage, loc: storage)
		model_scene2 = torch.load(opt.pre_trained_model_path_scene2, map_location=lambda storage, loc: storage)
		model_scene3 = torch.load(opt.pre_trained_model_path_scene3, map_location=lambda storage, loc: storage)
		model_scene4 = torch.load(opt.pre_trained_model_path_scene4, map_location=lambda storage, loc: storage)
		model_scene5 = torch.load(opt.pre_trained_model_path_scene5, map_location=lambda storage, loc: storage)

	model_scene1.eval()
	model_scene2.eval()
	model_scene3.eval()
	model_scene4.eval()
	model_scene5.eval()

	csv_reader = csv.reader(open("./data/val1_filename_id.csv"))
	file2id = {}
	final_json = {'results': [] }
	num = 0
	for row in csv_reader:
		file2id[row[0]] = row[1]
	for scene_num, scene in enumerate(sorted(os.listdir(val1_dir))):
		print('给场景%d加框：'%(scene_num + 1))
		if scene_num == 0:
			model = model_scene1
			opt.conf_threshold = 0.65
			opt.nms_threshold = 0.5

		elif scene_num == 1:
			model = model_scene2
			opt.conf_threshold = 0.6
			opt.nms_threshold = 0.4
		elif scene_num == 2:
			model = model_scene3
			opt.conf_threshold = 0.6
			opt.nms_threshold = 0.4
		elif scene_num == 3:
			model = model_scene4
			opt.conf_threshold = 0.6
			opt.nms_threshold = 0.6
		elif scene_num == 4:
			model = model_scene5
			opt.conf_threshold = 0.5
			opt.nms_threshold = 0.4

		scene_dir = os.path.join(val1_dir,scene)
		for p, img_dir in enumerate(sorted(os.listdir(scene_dir))):
			print('第%d张图片'%num)
			final_json["results"].append({"image_id": file2id[img_dir], "object":[]})
			img_dir2 = os.path.join(scene_dir,img_dir)
			predictions ,width_ratio ,height_ratio,width,height= yolo2(img_dir2,model)
			for pred in predictions:
				xmin = int(max(pred[0] / width_ratio, 0))
				ymin = int(max(pred[1] / height_ratio, 0))
				xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
				ymax = int(min((pred[1] + pred[3]) / height_ratio, height))

				if pred[5] == 'person' and not(xmax - xmin <50 and ymax - ymin <50):
					final_json["results"][num]["object"].append({"minx": int(xmin),
					                                             "miny": int(ymin),
					                                             "maxx": int(xmax),
					                                             "maxy": int(ymax),
					                                             "staff": -1, "customer": -1, "stand": -1, "sit": -1,
					                                             "play_with_phone": -1, "male": -1,
					                                             "female": -1, "confidence": 1})
				else:
					print('$$$$$$$$$$$$$$$$$$%s'%pred[5])
			num += 1

	with open(jsonout_dir, "w") as dump_f:
		json.dump(final_json, dump_f)
		print('save achieve')


def yolo2(image_path,model):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	height, width = image.shape[:2]
	image = cv2.resize(image, (opt.image_size, opt.image_size))
	image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
	image = image[None, :, :, :]
	width_ratio = float(opt.image_size) / width
	height_ratio = float(opt.image_size) / height
	data = Variable(torch.FloatTensor(image))
	if torch.cuda.is_available():
		data = data.cuda()
	with torch.no_grad():
		logits = model(data)
		predictions = post_processing(logits, opt.image_size, CLASSES, model.anchors, opt.conf_threshold,
		                              opt.nms_threshold)
	if len(predictions) != 0:
		predictions = predictions[0]

	return predictions ,width_ratio ,height_ratio,width,height


if __name__ == "__main__":
	opt = get_args()
	get_multiscene_result(opt,'./data/val1',"./result/test_jw927_1.json")
