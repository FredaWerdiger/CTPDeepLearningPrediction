import os
import glob
import numpy as np
import pandas as pd
import ants

# get list of segmented patients

HOMEDIR = os.path.expanduser('~/')

mediaflux = HOMEDIR + 'mediaflux/'
atlas = HOMEDIR + 'atlas/ATLAS_database/'

if not os.path.exists(mediaflux):
    mediaflux = 'X:/'
    atlas = 'Y:/ATLAS_database/'
directory = mediaflux + 'data_freda/ctp_project/CTP_DL_Data/'

# PREVIOUS STUDIES
# masks = os.path.join(directory, 'no_seg/masks')
# semi_data = os.path.join(directory, 'no_seg/masks_semi')
# masks_semi = [file for file in glob.glob(semi_data + '/*')
#               if ('exclude' not in file and "funny" not in file)]
# masks_semi_june = glob.glob(mediaflux  + 'data_freda/dl_seg_check/*seg*')
# out_location = os.path.join(directory, 'no_seg/masks_semi_ctp')
# out_dwi = os.path.join(directory, 'no_seg/dwi_ctp')

# out_dwi = os.path.join(directory, 'no_seg/dwi_ctp_june')
# out_location = os.path.join(directory, 'no_seg/masks_semi_ctp_june')

out_dwi = os.path.join(directory, 'no_reperfusion/dwi_inference/dwi_ctp')
out_location = os.path.join(directory, 'no_reperfusion/dwi_inference/segs_ctp')

# no reperfusion dataset
semi_data = os.path.join(directory, 'no_reperfusion/dwi_inference/segs')
masks_semi = glob.glob(semi_data + '/*')


if not os.path.exists(out_location):
    os.makedirs(out_location)

if not os.path.exists(out_dwi):
    os.makedirs(out_dwi)

# dwis = [file for file in glob.glob(mediaflux + 'data_freda/dl_seg_check/*') if ('pred.nii' not in file and 'seg.nii' not in file)]

masks_semi.sort()

reg = os.listdir(out_location)
dwi_reg = os.listdir(out_dwi)

subjects = ['INSP_' + os.path.basename(name).split('_')[1] for name in masks_semi]
subjects_reg = ['INSP_' + name.split('_')[1].split('.nii')[0] for name in dwi_reg]
subjects_not_done = [sub for sub in subjects if sub not in subjects_reg]
print(f"Number of subjects not done: {len(subjects_not_done)}")

for subject in subjects:
    # transform_matrix = glob.glob(atlas + subject + '/*/*/*/*.mat')
    # transform_files = glob.glob(atlas + subject + '/*/*/*/*Warped*')
    fixed = glob.glob(atlas + subject + '/CT_baseline/CTP_baseline/mistar/Mean_baseline/*mistar_brain.nii.gz')
    moving = glob.glob(atlas + subject + '/MR-follow_up/DWI-follow_up/*_b0.nii.gz')
    # moving_mask = glob.glob(semi_data + '/mask_' + subject + '.nii.gz')[0]
    moving_mask = [file for file in masks_semi if subject in file]
    try:
        moving_mask = moving_mask[0]
    except IndexError:
        moving_mask = [file for file in masks_semi if subject in file]
        try:
            moving_mask = moving_mask[0]
        except IndexError:
            try:
                moving_mask = glob.glob(
                    mediaflux + 'INSPIRE_database/' + subject + '/MR-follow_up/DWI-follow_up/*manual_lesion*')[0]
            except IndexError:
                print("Can't find semi mask for patient.")
    moving_dwi = glob.glob(atlas + subject + '/MR-follow_up/DWI-follow_up/*_b1000.nii.gz')
    brainmask = glob.glob(atlas + subject + '/CT_baseline/CTP_baseline/mistar/Mean_baseline/*mistar_brainmask.nii.gz')
    try:
        # transform_matrix[0]
        fixed = fixed[0]
    except IndexError:
        print("Missing mistar brain for patient {}".format(subject))
        print("Using BET brain...")
        try:
            fixed = glob.glob(
                atlas + subject + '/CT_baseline/CTP_baseline/mistar/Mean_baseline/*Mean_baseline_brain.nii.gz')[0]
            brainmask = glob.glob(
                atlas + subject + '/CT_baseline/CTP_baseline/mistar/Mean_baseline/*Mean_baseline_brainmask.nii.gz')
        except IndexError:
            print("Missing brain image {}".format(subject))
            try:
                fixed = glob.glob(
                    atlas + subject + '/CT_postIV/CTP_postIV/mistar/Mean_baseline/*Mean_baseline_brain.nii.gz')[0]
                brainmask = glob.glob(
                    atlas + subject + '/CT_postIV/CTP_postIV/mistar/Mean_baseline/*Mean_baseline_brainmask.nii.gz')
                print('Images are post IV')
            except IndexError:
                print('None found')
                continue
    try:
        moving = moving[0]
    except IndexError:
        print("Missing b0 for patient {}".format(subject))
        print("Using b1000 for registration...")
        try:
            moving = moving_dwi[0]
        except IndexError:
            print("Missing b1000 for patient {}".format(subject))
            # continue
    try:
        moving_dwi = moving_dwi[0]
        # transform_files = transform_matrix + transform_files
    except IndexError:
        print("Missing b1000 for patient {}".format(subject))
        # continue
    try:
        brainmask = brainmask[0]
    except IndexError:
        print('no brain mask')
        continue

    fixed = ants.image_read(fixed)
    moving = ants.image_read(moving)
    mask = ants.image_read(brainmask)
    fixed = fixed * mask
    print("Running transform for {}".format(subject))
    reg = ants.registration(
        fixed,
        moving,
        type_of_transform='Affine'
    )
    # transform lesion mask
    moving = moving_mask
    moving = ants.image_read(moving)
    fixedBrainReg = ants.apply_transforms(
        fixed, moving,
        reg['fwdtransforms'],
        interpolator='nearestNeighbor')
    if os.path.exists(os.path.join(out_location, 'mask_' + subject + '.nii.gz')):
        os.remove(os.path.join(out_location, 'mask_' + subject + '.nii.gz'))
    ants.image_write(fixedBrainReg * mask, os.path.join(out_location, 'mask_' + subject + '.nii.gz'))

    # transform dwi
    moving = moving_dwi
    moving = ants.image_read(moving)
    fixedBrainReg = ants.apply_transforms(
        fixed, moving,
        reg['fwdtransforms'],
        interpolator='bSpline')
    if os.path.exists(os.path.join(out_dwi, subject + '_dwi_ctp.nii.gz')):
        os.remove(os.path.join(out_dwi, subject + '_dwi_ctp.nii.gz'))
    ants.image_write(fixedBrainReg * mask, os.path.join(out_dwi, subject + '_dwi_ctp.nii.gz'))

    # for subject in subjects_not_done:
    #     fixed = glob.glob(atlas + subject + '/CT_baseline/CTP_baseline/mistar/Mean_baseline/*mistar_brain.nii.gz')
    #     mask = glob.glob(atlas + subject + '/CT_baseline/CTP_baseline/mistar/Mean_baseline/*mistar_brainmask.nii.gz')
    #     moving = glob.glob(atlas + subject + '/MR-follow_up/DWI-follow_up/*_b0.nii.gz')
    #     moving_dwi = glob.glob(atlas + subject + '/MR-follow_up/DWI-follow_up/*_b1000.nii.gz')
    #     try:
    #         # transform_matrix[0]
    #         fixed = fixed[0]
    #     except IndexError:
    #         print("Missing CTP image for patient {}".format(subject))
    #         continue
    #     try:
    #         moving = moving[0]
    #     except IndexError:
    #         print(f"Missing b0 image for patient {subject}")
    #         print("Using b1000 instead.")
    #         try:
    #             moving = moving_dwi[0]
    #         except IndexError:
    #             print("Missing b1000 image for patient {}".format(subject))
    #             continue
    #     try:
    #         moving_dwi = moving_dwi[0]
    #     except IndexError:
    #         print("Missing b1000 image for patient {}".format(subject))
    #         continue
    #     try:
    #         mask = mask[0]
    #     except IndexError:
    #         print("Missing mask image for patient {}".format(subject))
    #         continue
    #
    #         # transform_files = transform_matrix + transform_files
    #     if not os.path.exists(os.path.join(out_dwi, subject + '_dwi_ctp.nii.gz')):
    #         fixed = ants.image_read(fixed)
    #         moving = ants.image_read(moving)
    #         mask = ants.image_read(mask)
    #         print("Running transform for {}".format(subject))
    #         reg = ants.registration(
    #             fixed,
    #             moving,
    #             type_of_transform='Rigid'
    #         )
    #         moving = moving_dwi
    #         moving = ants.image_read(moving)
    #         fixedBrainReg = ants.apply_transforms(
    #             fixed, moving,
    #             reg['fwdtransforms'],
    #             interpolator='bSpline')
    #         ants.image_write(fixedBrainReg*mask, os.path.join(out_dwi, subject + '_dwi_ctp.nii.gz'))
