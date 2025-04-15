import os, glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from PIL import Image
import torchvision
from natsort import natsorted
from pycocotools.coco import COCO
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights, SSD300_VGG16_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import nms
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet18
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from collections import defaultdict
# from transforms import Transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision.utils as vutils
from torchvision.transforms import Pad
import torchvision.transforms.functional as F
from ultralytics import YOLO

model_label = "faster_rcnn_mobile_net"
def config_plot():
    '''
    Function to remove axis tickers and box around figure
    '''

    plt.box(False)
    plt.axis('off')

def setup_dataset_detection(image_dirpath, label_dirpath):
  '''
  Function to define train / val split
  Find the training samples as pairs: image/ground truth pair matching

  Assume directory structure of image_dirpath and label_dirpath is the same.
  data
   |-------image
   |         |-------sequence
   |                    |-------.png
   |-------bounding_box
             |-------sequence
                        |------.json


  Args:
    image_dirpath: string
      A path to the top level root directory containing all images
    label_dirpath: string
      A path to the top level root directory containing all labels

  Returns:
    train_image_paths: list[str]
      A list of strings representing the image paths of the train data
    train_label_paths: list[str]
      A list of strings representing the label paths of the train data
    test_image_paths: list[str]
      A list of strings representing the image paths of the test data
    test_label_paths: list[str]
      A list of strings representing the label paths of the test data
  '''
  # usually people do explicit data splits in industry i.e. these specific images go into train/val so if people change the structure/data it won't change your benchmarks
  # useful in industry
  test_set = ['test']

  image_paths = natsorted(glob.glob(os.path.join(image_dirpath, '*', '*.png')))

  label_paths = natsorted(glob.glob(os.path.join(label_dirpath, '*.json')))

  test_image_paths = []
  test_label_paths = []
  train_image_paths = []
  train_label_paths = []
  for image_path in image_paths:
    image_sequence_dirpath = image_path.split(os.sep)[-2]

    if image_sequence_dirpath in test_set:
      test_image_paths.append(image_path)
    else:
      train_image_paths.append(image_path)

  for label_path in label_paths:
    train_label_paths.append(label_path)
    test_label_paths.append(label_path)

  print(f"Test images: {len(test_image_paths)}", f"Train Images: {len(train_image_paths)}")

  return train_image_paths, train_label_paths, test_image_paths, test_label_paths


class CocoObjectDetectionDataset(data.Dataset):

    # initialise function of class
    def __init__(self, image_paths, label_paths):
      #list of tuples (name of parent directory for image, whole image path)
        self.image_paths = image_paths

        # the list of cocos
        #list of tuples (name of parent directory for image, coco)
        self.cocos = COCO(label_paths[0])

        #iterates through list of label paths, creates tuples (class, number of images in that category)
        self.elements_per_coco = [
            ('train', len(self.image_paths)),
        ]

        natsorted(self.elements_per_coco, key=lambda a: a[0])

        self.annotations = {}

        self.imageNames = []
        for image in image_paths:
          self.imageNames.append(image.split(os.sep)[-1])

        annotation_ids = []
        for k in self.cocos.anns.keys():
          image_ids = self.cocos.loadImgs(k)[0]
          if image_ids['file_name'] in self.imageNames:
            # annotation_ids.append(image_ids['id'])
            self.annotations[image_ids['file_name']] = self.cocos.imgToAnns[image_ids['id']]


    # obtain the sample with the given index
    def __getitem__(self, index):
        image_path = self.imageNames[index]

        if image_path in self.annotations:
          annotation = self.annotations[image_path]
        else:
          annotation = []

        image = Image.open(self.image_paths[index]).convert('RGB')
        num_objs = len(annotation)

        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = annotation[i]['bbox'][0]
            ymin = annotation[i]['bbox'][1]
            xmax = xmin + annotation[i]['bbox'][2]
            ymax = ymin + annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(annotation[i]['category_id'])

        if num_objs == 0:
          boxes = torch.zeros((0, 4), dtype=torch.float32)
          labels = torch.zeros((0,), dtype=torch.int64)
        else:
          boxes = torch.as_tensor(boxes, dtype=torch.float32)
          labels = torch.tensor(labels, dtype=torch.int64)

        image = torchvision.transforms.functional.to_tensor(image )

        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels

        return image, my_annotation, image_path

    # the total number of samples
    def __len__(self):
        return len(self.image_paths)

    def get_original_shape(self):
        return self.original_image_shape

def get_transform(size=480):
    #Ask in meeting: Do we need to normalize the dataset

    custom_transforms = []

    custom_transforms.append(torchvision.transforms.ToTensor())

    # if normalized, also use resize by uniform factor
    return torchvision.transforms.Compose(custom_transforms)

def get_test_transform(size = 480):
    custom_transforms = []

    custom_transforms.append(torchvision.transforms.ToTensor())

    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_detection(num_classes):
    # Load a Faster R-CNN model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one for our custom classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# def get_model_instance_detection(classes = 3, width= 640, height = 480):

#     model = torchvision.models.detection.ssd300_vgg16(
#         weights=SSD300_VGG16_Weights.COCO_V1
#     )

#     # Retrieve the list of input channels.
#     in_channels = _utils.retrieve_out_channels(model.backbone, (width, height))
#     # List containing number of anchors based on aspect ratios.
#     num_anchors = model.anchor_generator.num_anchors_per_location()
#     # The classification head.
#     model.head.classification_head = SSDClassificationHead(
#         in_channels=in_channels,
#         num_anchors=num_anchors,
#         num_classes=classes,
#



# def get_last_checkpoint(checkpoint_dirpath):

#   checkpoints = [f for f in os.listdir(checkpoint_dirpath)]
#   checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1].split('.')[0]) )

#   return os.path.join(checkpoint_dirpath, checkpoints[-1])

def get_last_checkpoint(checkpoint_dirpath):
    checkpoints = [f for f in os.listdir(checkpoint_dirpath) if f.startswith(model_label + '_epoch_') and f.endswith('.pth')]
    # print("checkpoints:", checkpoints)

    # Adjust the sorting key function to correctly extract the epoch number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('epoch_')[1].split('_loss')[0]))
    # print("checkpoints sorted", checkpoints)

    # Return the path to the latest checkpoint
    return os.path.join(checkpoint_dirpath, checkpoints[-1]) if checkpoints else None



def get_checkpoint_with_lowest_loss(checkpoint_dirpath):
    # Adjust the filter to match your filename pattern
    checkpoints = [f for f in os.listdir(checkpoint_dirpath) if f.startswith(model_label+'_epoch_') and f.endswith('.pth')]

    # Ensure the lambda function correctly extracts the floating-point loss value for comparison
    checkpoints = sorted(checkpoints, key=lambda x: float(x.split('_loss_')[1].replace('.pth', '')))

    # Return the path to the checkpoint with the lowest loss, if any checkpoints match the pattern
    return os.path.join(checkpoint_dirpath, checkpoints[0]) if checkpoints else None



def load_trained_model(checkpoint_path):
  print(checkpoint_path)
  model = get_model_instance_detection(2)
  checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
  module_state_dict = checkpoint_dict['net']

  model.load_state_dict(module_state_dict)

  return model


def iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def check_correctness(predictions, annotations, iou_threshold=0.5):
    predicted_scores = []
    outcomes = []  # 1 for correct, 0 for incorrect

    for pred_box, pred_label, pred_score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        correct = False
        for true_box, true_label in zip(annotations['boxes'], annotations['labels']):
            # Check if labels match and IoU exceeds threshold
            if pred_label == true_label and iou(pred_box, true_box) >= iou_threshold:
                correct = True
                break

        predicted_scores.append(pred_score.item())
        outcomes.append(int(correct))

    return predicted_scores, outcomes


def calc_MAP(dataloader, device, model, resize = 1000):
  model.eval()

  eval_predictions = []
  eval_annotations = []

  for step, (val_images, val_annotations, val_paths) in enumerate(dataloader):

          val_images = list(img.to(device) for img in val_images)
          val_annotations = [{k: v.to(device) for k, v in t.items()} for t in val_annotations]


          with torch.no_grad():
              predictions = model(val_images)


          for prediction, val_annotation in zip(predictions, val_annotations):
              boxes = prediction["boxes"]
              scores = prediction['scores']

              keep = nms(boxes.to('cuda'), scores.to('cuda'), iou_threshold=0.1)

              prediction['boxes'] = boxes[keep]
              prediction['labels'] = prediction['labels'][keep]
              prediction['scores'] = scores[keep]

              eval_predictions.append(prediction)
              eval_annotations.append(val_annotation)


  metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds = [0.5], class_metrics = True, )
  metric.update(eval_predictions, eval_annotations)
  result = metric.compute()
  return result

def pad_to_max_size(image, max_width, max_height):
    # Calculate padding for each side to match the maximum size
    left_pad = (max_width - image.shape[2]) // 2
    right_pad = max_width - image.shape[2] - left_pad
    top_pad = (max_height - image.shape[1]) // 2
    bottom_pad = max_height - image.shape[1] - top_pad

    # Apply padding
    return F.pad(image, (left_pad, top_pad, right_pad, bottom_pad), fill=0, padding_mode='constant')
