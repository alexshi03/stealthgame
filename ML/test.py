from functions import *
import torch
from torchvision.ops import nms
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    train_image_paths, train_label_paths, test_image_paths, test_label_paths = setup_dataset_detection('data/image', 'data/bounding_box')
    dataset_train = CocoObjectDetectionDataset(
        image_paths=train_image_paths,
        label_paths=train_label_paths)

    dataset_test = CocoObjectDetectionDataset(
        image_paths=test_image_paths,
        label_paths=test_label_paths)


    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn)

    fig, ax = plt.subplots(1)

    img, annotations, _ = dataset_train.__getitem__(0)

    img = img.permute(1, 2, 0)

    ax.imshow(img.numpy())

    for box in annotations['boxes']:
        width, height = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((box[0], box[1]), width, height, fill=False, edgecolor="purple", linewidth=2))

    plt.show()


    # stealth_dataset_train.__getitem__(0)