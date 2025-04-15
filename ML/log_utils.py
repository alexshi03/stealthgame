
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import math
from torchvision.ops import nms
def log_summary(writer, model_arch, learning_rate, batch_size, optimizer_type, scheduler, weight_decay):
    # Base tag for organization
    # base_tag = 'Experiment Parameters"

    summary = f'''Model Architecture: {model_arch},
    Learning Rate: {str(learning_rate)},
    Batch Size: {str(batch_size)},
    Optimizer: {optimizer_type},
    Scheduler: {scheduler},
    Weight Decay: {weight_decay}
    '''

    writer.add_text('Parameter Summary', summary, 0)


def log_stats(writer, train_map, val_map, learning_rate, epoch):
    writer.add_scalars('Train_Validation MAP_50_Global', {
        'Train': train_map,
        'Val': val_map
        }, epoch + 1)


def log_image_with_boxes(writer, num_images, images, annotations, model, device, epoch):
    model.eval()

    grid_size = int(math.sqrt(num_images))
    num_images = int(grid_size ** 2)

    images = images[0:num_images]
    annotations = annotations[0:num_images]

    images = list(img.to(device) for img in images)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

    with torch.no_grad():
        predictions = model(images)

    predictions_nms = []

    for prediction in predictions:
        boxes = prediction["boxes"]
        scores = prediction["scores"]

        keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.1)

        prediction["boxes"] = boxes[keep]
        prediction["labels"] = prediction["labels"][keep]
        prediction["scores"] = scores[keep]

        predictions_nms.append(prediction)

    #plotting the figure
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(25, 25), dpi=100) # Adjusting DPI here
    axs = axs.flatten()

    for ax_idx, ax in enumerate(axs):
        if ax_idx >= num_images:
            break

        image = images[ax_idx].permute(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())

        ax.imshow(image.cpu().numpy())
        ax.axis('off')

        color_dict_annotations = {
                1: "purple",
            }
        color_dict_predictions = {
            1: "blue",
        }
        #mapping the ground truths
        for box, label in zip(annotations[ax_idx]["boxes"], annotations[ax_idx]["labels"]):
            # box = box.cpu().numpy()
            box = box.cpu().numpy()
            label = label.item()
            width, height = box[2] - box[0], box[3] - box[1]
            ax.add_patch(plt.Rectangle((box[0], box[1]), width, height, fill=False, edgecolor=color_dict_annotations[label], linewidth=2))

        #mapping the predictions
        for box, label, score in zip(predictions_nms[ax_idx]["boxes"], predictions_nms[ax_idx]["labels"], predictions_nms[ax_idx]["scores"]):
            # box = box.cpu().numpy()
            # score = score.item()
            box = box.cpu().numpy()
            label = label.item()
            score = score.item()
            width, height = box[2] - box[0], box[3] - box[1]
            ax.add_patch(plt.Rectangle((box[0], box[1]), width, height, fill=False, edgecolor= color_dict_predictions[label], linewidth=2))
            ax.text(box[0], box[1] - 10, f'Pred: {score:.2f}', fontsize=12, color='red', verticalalignment='top')

    plt.tight_layout()

    # Add figure to tensorboard
    tag = f'Image Visualization/Epoch {epoch}'
    writer.add_figure(tag, fig, epoch)

    # Cleanup
    plt.close(fig)
