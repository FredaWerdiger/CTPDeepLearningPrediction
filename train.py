import random
import sys
from fns import *
sys.path.append('../DenseNetFCN3D-pytorch')
from densenet import *
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss
from torch.optim import Adam
import torch.nn.functional as f
from monai.metrics import DiceMetric
from monai.handlers.utils import from_engine
from monai.utils import set_determinism
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    Invertd,
    GaussianSmoothd,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RepeatChannelD,
    Resized,
    SaveImaged,
    ThresholdIntensityd,
    SplitDimd,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import f1_score, auc, roc_curve

'''
Input features

data_dir = location of the data. 

Data should be organised as folders as per organise_data.py
"images", "nccts" and "masks" for stacked CTP maps, contrast free CT and segmentation, respectively.

in addition hemisphere masks can be used for analysis of results and dwi images can be used for results pngs.

features = list of features to include in the model.
Features should be in this order: 'DT', 'CBF', 'CBV', 'MTT', 'ncct'.
e.g. ['DT', 'NCCT'] or ['CBF', 'MTT', 'NCCT']

out_tag = tag for the folder that will hold results. If left empty (out_tag='') the folder will be called "out"

test_cases = ''
a csv file with a list of cases that are denoted as test cases

notes = anything you want to add to the results print out.

'''


def main(data_dir, out_tag='', test_cases=None, features=None, image_size=None, max_epochs=None, notes=''):

    # ++++++++++++++++++++++
    # SET DEFAULT PARAMETERS
    # ++++++++++++++++++++++
    if features is None:
        features = ['DT', 'CBF', 'CBV', 'ncct']
    else:
        features = list(features)
    if image_size is None:
        image_size = 128
    image_size = [image_size]
    if max_epochs is None:
        max_epochs = 400

    image_paths = glob.glob(os.path.join(data_dir, 'images', '*'))
    image_paths.sort()
    mask_paths = glob.glob(os.path.join(data_dir, 'masks', '*'))
    mask_paths.sort()
    ncct_paths = glob.glob(os.path.join(data_dir, 'nccts', '*'))
    ncct_paths.sort()
    # below is only for creating results images after, optional
    dwi_paths = glob.glob(os.path.join(data_dir, 'dwis', '*'))
    dwi_paths.sort()
    # below are used to get results from one hemisphere
    # these are generated using https://github.com/FredaWerdiger/automatic_rotation
    left_hemisphere_masks = glob.glob(os.path.join(data_dir, 'left_hemisphere_mask', '*'))
    right_hemisphere_masks = glob.glob(os.path.join(data_dir, 'right_hemisphere_mask', '*'))

    # ENSURE DATA IS THERE
    assert image_paths is not None
    assert len(image_paths) == len(mask_paths) == len(ncct_paths)

    # CREATE LABEL AND DATAFRAME
    # labels should correspond to the image titles
    dl_id = [str(num) for num in np.arange(len(image_paths))]
    dl_id = [name.zfill(len(dl_id[-1])) for name in dl_id]

    df = pd.DataFrame(dl_id, columns=['dl_id'])
    # save the image paths associated with the labels for reference
    df['image_paths'] = image_paths
    df['mask_paths'] = mask_paths
    df['dwi_paths'] = dwi_paths

    # +++++++++++++++
    # READ TEST CASES
    # +++++++++++++++
    if test_cases:
        extra_test_id = pd.read_csv(
            test_cases,
            sep=',',
            header=None,
            names=['dl_id']).dl_id.to_list()
        extra_test_id = [str(name).zfill(len(dl_id[-1])) for name in extra_test_id]
        dl_id = [name for name in dl_id if name not in extra_test_id]
    # +++++++++++++++++++++++
    # STRATIFY BY LESION SIZE
    # +++++++++++++++++++++++

    random_state = 42
    lesion_size = []
    for path in mask_paths:
        im = sitk.ReadImage(path)
        x, y, z = im.GetSpacing()
        voxel_size = (x * y * z) / 1000
        label = sitk.LabelShapeStatisticsImageFilter()
        label.Execute(sitk.Cast(im, sitk.sitkUInt8))
        size = label.GetNumberOfPixels(1)
        lesion_size.append(voxel_size * size)

    # SMALL LESIONS DEFINED AS LESS THAN 5ML
    labels = (np.asarray(lesion_size) < 5) * 1
    df['size_labels'] = labels
    labels = df[df.apply(lambda x: x.dl_id in dl_id, axis=1)].size_labels.to_list()

    num_train = int(np.ceil(0.6 * len(labels)))
    num_validation = int(np.ceil(0.2 * len(labels)))
    num_test = len(labels) - (num_train + num_validation)

    train_id, test_id = train_test_split(dl_id,
                                         train_size=num_train,
                                         test_size=num_test + num_validation,
                                         random_state=random_state,
                                         shuffle=True,
                                         stratify=labels)

    test_df = df[df.apply(lambda x: x['dl_id'] in test_id, axis=1)]
    test_labels = test_df.size_labels.to_list()
    test_id = test_df.dl_id.to_list()

    validation_id, test_id = train_test_split(test_id,
                                              train_size=num_validation,
                                              test_size=num_test,
                                              random_state=random_state,
                                              shuffle=True,
                                              stratify=test_labels)

    if test_cases:
        test_id = test_id + extra_test_id
    # GET THE NUMBER OF SMALL LESIONS IN EACH GROUP
    # LABEL IDS
    df['group'] = ''
    train_df = df[df.apply(lambda x: x.dl_id in train_id, axis=1)]
    num_small_train = len(train_df[train_df.apply(lambda x: x.size_labels == 1, axis=1)])
    for id in train_id:
        df.loc[df.dl_id == id, 'group'] = 'train'

    val_df = df[df.apply(lambda x: x.dl_id in validation_id, axis=1)]
    num_small_val = len(val_df[val_df.apply(lambda x: x.size_labels == 1, axis=1)])
    for id in validation_id:
        df.loc[df.dl_id == id, 'group'] = 'validation'

    test_df = df[df.apply(lambda x: x.dl_id in test_id, axis=1)]
    num_small_test = len(test_df[test_df.apply(lambda x: x.size_labels == 1, axis=1)])
    for id in test_id:
        df.loc[df.dl_id == id, 'group'] = 'test'

    # +++++++++++++
    # BUILD DATASET
    # +++++++++++++

    def make_dict(id):
        id = [str(num).zfill(3) for num in id]
        paths1 = [file for file in image_paths
                  if file.split('.nii.gz')[0].split('_')[-1] in id]
        paths2 = [file for file in ncct_paths if file.split('.nii.gz')[0].split('_')[-1] in id]
        paths3 = [file for file in mask_paths if file.split('.nii.gz')[0].split('_')[-1] in id]

        files_dict = [{"image": image_name, "ncct": ncct_name, "label": label_name} for
                      image_name, ncct_name, label_name in zip(paths1, paths2, paths3)]

        return files_dict

    train_files = make_dict(train_id)
    val_files = make_dict(validation_id)
    test_files = make_dict(test_id)

    features_transform = ['image_' + string for string in [feature for feature in features
                                                           if "ncct" not in feature]]
    if 'ncct' in features:
        features_transform += ['ncct_raw']
    features_string = ''
    for feature in features:
        features_string += '_'
        features_string += feature
    patch_size = None
    batch_size = 2
    val_interval = 2

    print(f"out_tag = {out_tag}")

    if not os.path.exists(os.path.join(data_dir, 'out_' + out_tag)):
        os.makedirs(os.path.join(data_dir, 'out_' + out_tag))

    set_determinism(seed=42)

    transform_dir = os.path.join(data_dir, 'train', 'ncct_trans')
    if not os.path.exists(transform_dir):
        os.makedirs(transform_dir)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "ncct", "label"]),
            EnsureChannelFirstd(keys=["image", "ncct", "label"]),
            Resized(keys=["image", "ncct", "label"],
                    mode=['trilinear', 'trilinear', "nearest"],
                    align_corners=[True, True, None],
                    spatial_size=image_size * 3),
            SplitDimd(keys="image", dim=0, keepdim=True,
                      output_postfixes=['DT', 'CBF', 'CBV', 'MTT']),
            RepeatChannelD(keys="ncct", repeats=2),
            SplitDimd(keys="ncct", dim=0, keepdim=True,
                      output_postfixes=['raw', 'atrophy']),
            ThresholdIntensityd(keys="ncct_atrophy", threshold=15, above=False),
            ThresholdIntensityd(keys="ncct_atrophy", threshold=0, above=True),
            GaussianSmoothd(keys="ncct_atrophy", sigma=1),
            ConcatItemsd(keys=features_transform, name="image", dim=0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=1.0),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "ncct", "label"]),
            EnsureChannelFirstd(keys=["image", "ncct", "label"]),
            Resized(keys=["image", "ncct", "label"],
                    mode=['trilinear', 'trilinear', "nearest"],
                    align_corners=[True, True, None],
                    spatial_size=image_size * 3),
            SplitDimd(keys="image", dim=0, keepdim=True,
                      output_postfixes=["DT", "CBF", "CBV", "MTT"]),
            RepeatChannelD(keys="ncct", repeats=2),
            SplitDimd(keys="ncct", dim=0, keepdim=True,
                      output_postfixes=['raw', 'atrophy']),
            ThresholdIntensityd(keys="ncct_atrophy", threshold=15, above=False),
            ThresholdIntensityd(keys="ncct_atrophy", threshold=0, above=True),
            GaussianSmoothd(keys="ncct_atrophy", sigma=1),
            ConcatItemsd(keys=features_transform, name="image", dim=0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "ncct", "label"]),
            EnsureChannelFirstd(keys=["image", "ncct", "label"]),
            Resized(keys=["image", "ncct"],
                    mode=['trilinear', 'trilinear'],
                    align_corners=[True, True],
                    spatial_size=image_size * 3),
            SplitDimd(keys="image", dim=0, keepdim=True,
                      output_postfixes=['DT', 'CBF', 'CBV', 'MTT']),
            RepeatChannelD(keys="ncct", repeats=2),
            SplitDimd(keys="ncct", dim=0, keepdim=True,
                      output_postfixes=['raw', 'atrophy']),
            ThresholdIntensityd(keys="ncct_atrophy", threshold=15, above=False),
            ThresholdIntensityd(keys="ncct_atrophy", threshold=0, above=True),
            GaussianSmoothd(keys="ncct_atrophy", sigma=1),
            ConcatItemsd(keys=features_transform, name="image", dim=0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    train_dataset = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=8)

    val_dataset = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=8)

    test_ds = CacheDataset(
        data=test_files,
        transform=test_transforms,
        cache_rate=1.0,
        num_workers=8
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            pin_memory=True)

    test_loader = DataLoader(test_ds,
                             batch_size=1,
                             pin_memory=True)

    # ++++++++++++
    # SANITY CHECK
    # ++++++++++++

    # UNCOMMENT EVERYTHING BELOW FOR SANITY CHECK
    # EACH RUN GENERATES A DIFFERENT RANDOM SLICE OF A DIFFERENT RANDOM IMAGE
    m = random.randint(0, len(train_files) - 1)
    data_example = train_dataset[m]
    ch_in = data_example['image'].shape[0] # used for model input information
    # s = random.randint(20, image_size[0] - 20)
    # plt.figure("image", (18, 4))
    # for i in range(ch_in):
    #     plt.subplot(1, ch_in + 1, i + 1)
    #     plt.title(f"image channel {i}")
    #     plt.imshow(data_example["image"][i, :, :, s].detach().cpu(), cmap="jet")
    #     plt.axis('off')
    # # also visualize the 3 channels label corresponding to this image
    # print(f"label shape: {data_example['label'].shape}")
    # plt.subplot(1, 6, 6)
    # plt.title("label")
    # plt.imshow(data_example["label"][0, :, :, s].detach().cpu(), cmap="gray")
    # plt.axis('off')
    # plt.show()
    # plt.close()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    channels = (16, 32, 64)

    model = AttentionUnet(
        spatial_dims=3,
        in_channels=ch_in,
        out_channels=2,
        channels=channels,
        strides=(2, 2, 2),
    )

    model.to(device)

    loss_function = DiceCELoss(smooth_dr=1e-5,
                               smooth_nr=0,
                               to_onehot_y=True,
                               softmax=True,
                               include_background=False,
                               squared_pred=True,
                               lambda_dice=1,
                               lambda_ce=1)

    learning_rate = 1e-4
    optimizer = Adam(model.parameters(),
                     learning_rate,
                     weight_decay=1e-5)

    dice_metric = DiceMetric(include_background=False, reduction='mean')
    dice_metric_train = DiceMetric(include_background=False, reduction='mean')

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    epoch_loss_values = []
    dice_metric_values = []
    dice_metric_values_train = []
    best_metric = -1
    best_metric_epoch = -1

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    start = time.time()
    # SAVING THE MODEL DURING TRAINING
    model_path = 'best_metric_' + model._get_name() + '_' + str(max_epochs) + '_' + features_string + '.pth'

    # ++++++++++++++++
    # TRAIN THE MODEL
    # ++++++++++++++++

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        epoch_loss = 0
        step = 0
        model.train()
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            print("Evaluating...")
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = model(val_inputs)
                    # compute metric for current iteration
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(val_outputs, val_labels)

                mean_dice = dice_metric.aggregate().item()
                dice_metric.reset()
                dice_metric_values.append(mean_dice)

                # repeating the process for training data to check for overfitting
                for val_data in train_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = model(val_inputs)

                    # compute metric for current iteration
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric_train(val_outputs, val_labels)

                mean_dice_train = dice_metric_train.aggregate().item()
                dice_metric_train.reset()
                dice_metric_values_train.append(mean_dice_train)

                if mean_dice > best_metric:
                    best_metric = mean_dice
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        data_dir, 'out_' + out_tag, model_path))
                    print("saved new best metric model")

                print(
                    f"current epoch: {epoch + 1} current mean dice: {mean_dice:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
        del loss, outputs
    end = time.time()
    time_taken = end - start
    print(f"Time taken: {round(time_taken, 0)} seconds")
    time_taken_hours = time_taken/3600
    time_taken_mins = np.ceil((time_taken/3600 - int(time_taken/3600)) * 60)
    time_taken_hours = int(time_taken_hours)

    model_name = model._get_name()
    loss_name = loss_function._get_name()

    # ++++++++++++++++++++++++++++++
    # SAVE INFORMATION ABOUT RESULTS
    # ++++++++++++++++++++++++++++++

    # SAVE A TEXT FILE WITH ESSENTIAL INFORMATION FOR REPRODUCTION
    with open(
            data_dir + 'out_' + out_tag + '/model_info_' + str(max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_' + features_string + '.txt', 'w') as myfile:
        myfile.write(f'Train dataset size: {len(train_files)}\n')
        myfile.write(f'Number of lesions under 5mL: {num_small_train}\n')
        myfile.write(f'Validation dataset size: {len(val_files)}\n')
        myfile.write(f'Number of lesions under 5mL: {num_small_val}\n')
        myfile.write(f'Test dataset size: {len(test_files)}\n')
        myfile.write(f'Number of lesions under 5mL: {num_small_test}\n')
        myfile.write(f'Intended number of features: {len(features)}\n')
        myfile.write(f'Actual number of features: {ch_in}\n')
        myfile.write('Features: ')
        myfile.write(features_string)
        myfile.write('\n')
        myfile.write(f'Model: {model_name}\n')
        myfile.write(f'Loss function: {loss_name}\n')
        myfile.write(f'Initial Learning Rate: {learning_rate}\n')
        myfile.write(f'Number of epochs: {max_epochs}\n')
        myfile.write(f'Batch size: {batch_size}\n')
        myfile.write(f'Image size: {image_size}\n')
        myfile.write(f'Patch size: {patch_size}\n')
        myfile.write(f'channels: {channels}\n')
        myfile.write(f'Validation interval: {val_interval}\n')
        myfile.write(f"Best metric: {best_metric:.4f}\n")
        myfile.write(f"Best metric epoch: {best_metric_epoch}\n")
        myfile.write(f"Time taken: {time_taken_hours} hours, {time_taken_mins} mins\n")
        myfile.write(notes)

    # PLOT TRAINING AND VALIDATION LOSS PER EPOCH
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Average Loss per Epoch")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Mean Dice (Accuracy)")
    x = [val_interval * (i + 1) for i in range(len(dice_metric_values))]
    y = dice_metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, 'b', label="Dice on validation data")
    y = dice_metric_values_train
    plt.plot(x, y, 'k', label="Dice on training data")
    plt.legend(loc="center right")
    plt.savefig(os.path.join(data_dir + 'out_' + out_tag,
                             'loss_plot_' + str(max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_' + features_string +'.png'),
                bbox_inches='tight', dpi=300, format='png')
    plt.close()

    # +++++++++++++++++++++++++
    # INFERENCE ON THE TEST SET
    # +++++++++++++++++++++++++

    # location to save binary predictions (50% probability threshold)
    pred_dir = os.path.join(data_dir, 'out_' + out_tag, "pred_" + features_string)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    # location to save probability maps
    proba_dir = os.path.join(data_dir, 'out_' + out_tag, "proba_" + features_string)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    # location to save images for display
    png_dir = os.path.join(data_dir, 'out_' + out_tag, "proba_pngs_" + features_string)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    post_transforms = Compose([
        EnsureTyped(keys=["pred", "label"]),
        Invertd(
            keys=["pred", "proba"],
            transform=test_transforms,
            orig_keys=["image", "image"],
            meta_keys=["pred_meta_dict", "pred_meta_dict"],
            orig_meta_keys=["image_meta_dict", "image_meta_dict"],
            meta_key_postfix="meta_dict",
            nearest_interp=[False, False],
            to_tensor=[True, True],
        ),
        AsDiscreted(keys="label", to_onehot=2),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        SaveImaged(
            keys="proba",
            meta_keys="pred_meta_dict",
            output_dir=proba_dir,
            output_postfix="proba",
            resample=False,
            separate_folder=False),
        SaveImaged(
            keys="pred",
            meta_keys="pred_meta_dict",
            output_dir=pred_dir,
            output_postfix="seg",
            resample=False,
            separate_folder=False)
    ])

    loader = LoadImage(image_only=True)
    loader_meta = LoadImage(image_only=False)

    model.load_state_dict(torch.load(os.path.join(data_dir, 'out_' + out_tag, model_path)))

    model.eval()

    results = pd.DataFrame(columns=['dl_id',
                                    'dice',
                                    'hemisphere',
                                    'dice70',
                                    'dice90',
                                    'auc',
                                    'auc70',
                                    'auc90',
                                    'sensitivity',
                                    'sensitivity70',
                                    'sensitivity90',
                                    'specificity',
                                    'specificity70',
                                    'specificity90',
                                    'ppv',
                                    'ppv70',
                                    'ppv90',
                                    'npv',
                                    'npv70',
                                    'npv90',
                                    'size',
                                    'size_pred',
                                    'size_ml',
                                    'size_pred_ml'])

    test_id.sort()
    results['dl_id'] = test_id

    dice_metric = []
    dice_metric70 = []
    dice_metric90 = []
    sensitivities = []
    specificities = []
    gts_flat = []
    preds_flat = []

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs = test_data["image"].to(device)
            test_data["pred"] = model(test_inputs)
            # GENERATE THE PROBABILITY OF INFARCT
            prob = f.softmax(test_data["pred"], dim=1)
            test_data["proba"] = prob
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
            test_output, test_label, test_image, test_proba = from_engine(
                ["pred", "label", "image", "proba"])(test_data)
            # GET THE PATH OF THE ORIGINAL IMAGE FILE
            original_path = test_data[0]["image_meta_dict"]["filename_or_obj"][0]
            original_image = loader_meta(original_path)
            # GET THE SIZE OF THE PIXEL FOR CALCULATING ABSOLUTE VOLUME
            volx, voly, volz = original_image[1]['pixdim'][1:4]
            pixel_vol = volx * voly * volz

            ground_truth = test_label[0][1].detach().numpy()
            # GET THE BINARY PREDICTION FROM EACH PROBABILITY BAND MAP 50/70/90
            prediction = (test_proba[0][1].detach().numpy() >= 0.5) * 1
            prediction_70 = (test_proba[0][1].detach().numpy() >= 0.7) * 1
            prediction_90 = (test_proba[0][1].detach().numpy() >= 0.9) * 1
            # get the name of the subject - dl_id - based on the image path
            name = df[df.apply(lambda x:
                                  os.path.basename(x.image_paths) == os.path.basename(original_path),
                                  axis=1)].dl_id.values[0]
            # get the hemisphere masks associated with each subject
            hemisphere_mask = None
            try:
                left_mask = [file for file in left_hemisphere_masks if name in file][0]
                right_mask = [file for file in right_hemisphere_masks if name in file][0]
                left_im, right_im = [loader(im) for im in [left_mask, right_mask]]
                left_np, right_np = [im.detach().numpy() for im in [left_im, right_im]]

                # find which hemisphere the lesion is in
                right_masked = right_np * ground_truth
                left_masked = left_np * ground_truth

                # see if there are any pixels in each corner
                counts_right = np.count_nonzero(right_masked)
                counts_left = np.count_nonzero(left_masked)
                if counts_right > counts_left:
                    hemisphere_mask = right_np.flatten()
                    results.loc[results.id == name, 'hemisphere'] = 'right'
                elif counts_right < counts_left:
                    hemisphere_mask = left_np.flatten()
                    results.loc[results.id == name, 'hemisphere'] = 'left'
                else:
                    # is this case there is no lesion, but still a prediction?
                    right_masked = right_np * prediction
                    left_masked = left_np * prediction
                    counts_right = np.count_nonzero(right_masked)
                    counts_left = np.count_nonzero(left_masked)
                    if counts_right > counts_left:
                        hemisphere_mask = right_np.flatten()
                        results.loc[results.id == name, 'hemisphere'] = 'right'
                    else:
                        hemisphere_mask = left_np.flatten()
                        results.loc[results.id == name, 'hemisphere'] = 'left'
            except IndexError:
                print('No hemisphere mask, evaluating across whole brain')
            gt_flat = ground_truth.flatten()
            pred_flat = prediction.flatten()
            pred70_flat = prediction_70.flatten()
            pred90_flat = prediction_90.flatten()
            dice_score = f1_score(gt_flat, pred_flat)
            dice_metric.append(dice_score)
            dice70 = f1_score(gt_flat, pred70_flat)
            dice_metric70.append(dice70)
            dice90 = f1_score(gt_flat, pred90_flat)
            dice_metric90.append(dice90)
            print(f"Dice score for image: {dice_score:.4f}")
            if hemisphere_mask is not None:
                pred_flat = np.where((hemisphere_mask == 0), np.nan, pred_flat)
                gt_flat = np.where((hemisphere_mask == 0), np.nan, gt_flat)
                core_flat = np.where(hemisphere_mask == 0, np.nan, pred_flat)
            preds_flat.extend(pred_flat)
            gts_flat.extend(gt_flat.astype(int))
            tp = len(np.where((gt_flat == 1) & (core_flat == 1))[0])
            fp = len(np.where((gt_flat == 0) & (core_flat == 1))[0])
            fn = len(np.where((gt_flat == 1) & (core_flat == 0))[0])
            tn = len(np.where((gt_flat == 0) & (core_flat == 0))[0])
            if (tp == 0) and (fn == 0):
                sensitivity = 1
            else:
                sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            if (tp == 0) and (fp == 0):
                if fn == 0:
                    ppv = 1
                else:
                    ppv = 0
            else:
                ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            # mask out nans and recalculate AUC
            fpr, tpr, threshold = roc_curve(gt_flat[np.where((gt_flat == 1) | (gt_flat == 0))],
                                            core_flat[np.where((core_flat == 1) | (core_flat == 0))])
            auc_score = auc(fpr, tpr)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            # do the same for other thresholds
            # 70
            core_flat = np.where(hemisphere_mask == 0, np.nan, pred70_flat)
            tp = len(np.where((gt_flat == 1) & (core_flat == 1))[0])
            fp = len(np.where((gt_flat == 0) & (core_flat == 1))[0])
            fn = len(np.where((gt_flat == 1) & (core_flat == 0))[0])
            tn = len(np.where((gt_flat == 0) & (core_flat == 0))[0])

            # ppv and false emission rates
            if (tp == 0) and (fn == 0):
                sensitivity70 = 1
            else:
                sensitivity70 = tp / (tp + fn)
            specificity70 = tn / (tn + fp)
            if (tp == 0) and (fp == 0):
                if fn == 0:
                    ppv70 = 1
                else:
                    ppv70 = 0
            else:
                ppv70 = tp / (tp + fp)
            npv70 = tn / (tn + fn)
            # mask out nans and recalculate AUC
            fpr, tpr, threshold = roc_curve(gt_flat[np.where((gt_flat == 1) | (gt_flat == 0))],
                                            core_flat[np.where((core_flat == 1) | (core_flat == 0))])
            auc_score70 = auc(fpr, tpr)
            # 90
            core_flat = np.where(hemisphere_mask == 0, np.nan, pred90_flat)
            tp = len(np.where((gt_flat == 1) & (core_flat == 1))[0])
            fp = len(np.where((gt_flat == 0) & (core_flat == 1))[0])
            fn = len(np.where((gt_flat == 1) & (core_flat == 0))[0])
            tn = len(np.where((gt_flat == 0) & (core_flat == 0))[0])
            if (tp == 0) and (fn == 0):
                sensitivity90 = 1
            else:
                sensitivity90 = tp / (tp + fn)
            specificity90 = tn / (tn + fp)
            if (tp == 0) and (fp == 0):
                if fn == 0:
                    ppv90 = 1
                else:
                    ppv90 = 0
            else:
                ppv90 = tp / (tp + fp)
            npv90 = tn / (tn + fn)
            # mask out nans and recalculate AUC
            fpr, tpr, threshold = roc_curve(gt_flat[np.where((gt_flat == 1) | (gt_flat == 0))],
                                            core_flat[np.where((core_flat == 1) | (core_flat == 0))])
            auc_score90 = auc(fpr, tpr)

            # volume
            size = ground_truth.sum()
            size_ml = size * pixel_vol / 1000

            size_pred = prediction.sum()
            size_pred_ml = size_pred * pixel_vol / 1000

            # For png images, optional
            try:
                dwi_img = [file for file in dwi_paths if name in file][0]
                dwi_img = loader(dwi_img)
                # below was necessary on spartan
                dwi_img = dwi_img.detach().numpy()
                save_loc = png_dir + '/' + name + '_proba.png'
                create_dwi_ctp_proba_map(dwi_img, ground_truth, prediction, prediction_70, prediction_90, save_loc,
                                         define_zvalues(dwi_img), ext='png', save=True)
            except IndexError:
                print("no_dwi_image, not generating png")
            # populate the results dataframe
            results.loc[results.id == name, 'size'] = size
            results.loc[results.id == name, 'size_ml'] = size_ml
            results.loc[results.id == name, 'size_pred'] = size_pred
            results.loc[results.id == name, 'size_pred_ml'] = size_pred_ml
            results.loc[results.id == name, 'dice'] = dice_score
            results.loc[results.id == name, 'dice70'] = dice70
            results.loc[results.id == name, 'dice90'] = dice90
            results.loc[results.id == name, 'auc'] = auc_score
            results.loc[results.id == name, 'auc70'] = auc_score70
            results.loc[results.id == name, 'auc90'] = auc_score90
            results.loc[results.id == name, 'sensitivity'] = sensitivity
            results.loc[results.id == name, 'sensitivity70'] = sensitivity70
            results.loc[results.id == name, 'sensitivity90'] = sensitivity90
            results.loc[results.id == name, 'specificity'] = specificity
            results.loc[results.id == name, 'specificity70'] = specificity70
            results.loc[results.id == name, 'specificity90'] = specificity90
            results.loc[results.id == name, 'ppv'] = ppv
            results.loc[results.id == name, 'ppv70'] = ppv70
            results.loc[results.id == name, 'ppv90'] = ppv90
            results.loc[results.id == name, 'npv'] = npv
            results.loc[results.id == name, 'npv70'] = npv70
            results.loc[results.id == name, 'npv90'] = npv90
        # aggregate the final mean dice result
        metric = np.mean(dice_metric)
        metric70 = np.mean(dice_metric70)
        metric90 = np.mean(dice_metric90)
        metric_recall = np.mean(sensitivities)
        metric_specificity = np.mean(specificities)
        metric_auc = np.mean(auc_score)
    print(f"Mean dice on test set: {metric:.4f}")
    # populate results dataframe with mean values
    results['mean_dice'] = metric
    results['mean_dice_70'] = metric70
    results['mean_dice_90'] = metric90
    results['mean_sensitvity'] = metric_recall
    results['mean_specificity'] = metric_specificity
    results['mean_auc'] = metric_auc

    results.to_csv(
        os.path.join(data_dir, 'out_' + out_tag, 'results_' + str(
            max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_' + features_string + '.csv'), index=False)
    df.to_csv(
        os.path.join(data_dir, 'out_' + out_tag, 'train_test_split.csv'), index=False
    )
    # MAKE AUC CURVE (only three points for binary output data)
    fpr, tpr, threshold = roc_curve(gts_flat, preds_flat)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - Specificity')
    plt.savefig(os.path.join(data_dir + 'out_' + out_tag,
                             'roc_plot_' + str(
                                 max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_' + features_string + '.png'),
                bbox_inches='tight', dpi=300, format='png')
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Must specify location of data.')
    else:
        main(*sys.argv[1:])
