import os
import torch
from functions import *

from torch.utils.tensorboard import SummaryWriter
from log_utils import *

import datetime

'''
Define parameters to be used in the rest of the model:
'''
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.001
EPOCHS = 30
GRAYSCALE = False
OPTIMIZER = "Adam"
SCHEDULER = "ReduceLRonPlateau"

'''
probability=0.5,
resize=600,
cropy=300,
cropx=300,
pad_left=100,
pad_bottom=100,
pad_right=200,
pad_top=200,
brightness_minmax = [0.5, 1.5],
saturation_minmax = [0.5, 1.5],
contrast_minmax = [0.5, 1.5],
hue_minmax = [-0.1, 0.1],
gaussian_blur_kernel_sizes = [3, 5, 7, 9],
gaussian_noise_sigma = 0.01
'''


#new option: using stepLR, don't use plteau
learning_rates = []

#hardcode learning rates

MODEL_ARCH = "Faster_RCNN_MobileNet"
log_dir = "runs/experiment_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")



model_label = "faster_rcnn_mobile_net"

if __name__ == "__main__":
    writer = SummaryWriter(log_dir)
    log_summary(writer, MODEL_ARCH, LEARNING_RATE, BATCH_SIZE, OPTIMIZER, SCHEDULER, WEIGHT_DECAY)
    train_batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE

    # 1 class: player, + background
    num_classes = 2

    #loading in images and labels
    train_image_paths, \
    train_label_paths, \
    test_image_paths, \
    test_label_paths = setup_dataset_detection('data/image', 'data/bounding_box')

    #sets up the COCO dataset instance
    dataset_train = CocoObjectDetectionDataset(
        image_paths=train_image_paths,
        label_paths=train_label_paths)

    dataset_test = CocoObjectDetectionDataset(
        image_paths=test_image_paths,
        label_paths=test_label_paths)


    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn)

    model = get_model_instance_detection(num_classes)

    # move model to the right device and set it to train
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model.to(device)
    print(next(model.parameters()).device)
    # model.train()

    # Initialize optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=250, mode='triangular2', base_momentum = 0.85, max_momentum = 0.99)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=250, mode='triangular2')

    checkpoint_folder = "checkpoints" +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(checkpoint_folder , exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, model_label+ '_epoch_{}_loss_{:.4f}_val_map_{:4f}.pth')

    min_loss = float('inf')
    max_val_map = 0
    for epoch in range(0, EPOCHS):
        model.train()
        print(f"Epoch {epoch + 1}/{EPOCHS}:")
        total_loss = 0.0  # Initialize total loss for the epoch
        num_steps = 0  # Count the number of steps in this epoch
        sum_loss_dict = {}
        log_images= []
        log_annotations = []
        for step, (images, annotations, image_path) in enumerate(data_loader):

            # images, annotations = train_transforms.transform(images, annotations, 0.5, resize =resize)

            if epoch % 10 == 0:
                log_images.extend(images)
                log_annotations.extend(annotations)
            #moving images and annotations to the GPU

            images = list(img.to(device) for img in images)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]


            loss_dict = model(images, annotations)
            losses = sum(loss for loss in loss_dict.values())
            for key in loss_dict:
                if key not in sum_loss_dict:
                    sum_loss_dict[key] = loss_dict[key]
                else:
                    sum_loss_dict[key] += loss_dict[key]

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            writer.add_scalar('Training Loss per Step', losses, epoch * len(data_loader) + step)

            num_steps += 1


        if epoch % 10 == 0:
            #plotting images
            log_image_with_boxes(writer, 16, log_images, log_annotations, model, device, epoch + 1)

        writer.add_scalar('Training Loss per Epoch', losses, epoch + 1)

        avg_loss = total_loss / num_steps
        avg_loss_dict = {}

        for key in sum_loss_dict:
            avg_loss_dict[key] = sum_loss_dict[key]/num_steps
        print(f"Epoch {epoch + 1}, Completed- Average Total Loss: {avg_loss:.4f}; Loss Breakdown {avg_loss_dict}")

        val_result = calc_MAP(dataloader = data_loader_test, device = device, model = model)

        print(" ")
        print("Validation MAP")
        for key in val_result:
            print(key, ": ", val_result[key])

        val_map = val_result['map']

        train_result = calc_MAP(dataloader = data_loader, device = device, model = model)

        print(" ")
        print("Train MAP")
        for key in train_result:
            print(key, ": ", train_result[key])

        train_map = train_result['map']

        learn_rate = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', learn_rate, epoch + 1)


        log_stats(writer,
                train_map,
                val_map,
                learn_rate,
                epoch+1)



        # should I be changing up this scheduler now?
        scheduler.step(val_map)


        state_dict = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step }


        max_val_map = val_map
        checkpoint_save = checkpoint_path.format(epoch, avg_loss, val_map)
        torch.save(state_dict, checkpoint_save)
        writer.add_text('Checkpoint Path', checkpoint_save, epoch + 1)
        print(checkpoint_save)
        # min_loss = avg_loss
