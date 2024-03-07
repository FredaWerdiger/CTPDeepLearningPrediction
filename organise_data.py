"""
Arranging data from ATLAS database for deep learning
Data has already been quality checked
CTP maps need to be stacked into a multi-channel feature map, and places into
DWI lesion (coregistered to CTP) needs to be recast as ground truth
"""
import os
import pandas as pd
from random import seed, sample
import shutil
import nibabel as nb
import numpy as np
import glob
import SimpleITK as sitk
from scipy.ndimage import morphology
import time
import sys

# https://github.com/FredaWerdiger/automatic_rotation
path_to_hemisphere_masking = '../automatic_rotation'  # change if needed
sys.path.append(path_to_hemisphere_masking)
import infer

'''
Arguments:
subject_file:
subject file input is expected to be comma separated values file of INSPIRE subject names
created like
with open('test_file.csv', 'w') as myfile:
    writer = csv.writer(myfile)
    for subject in subs:
        writer.writerow([subject])
atlas_path:
The location of the mediaflux drive (e.g. Y:)
out_path:
the location of the result. Within the specified path, the stacked images will be saved under 'images'
'''


def main(subjects_file, atlas_path, mediaflux_path, out_path, overwrite=False):
    # +++++++++++++++++++++++++
    # CREATE OUTPUT DIRECTORIES
    # +++++++++++++++++++++++++

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    sub_folder = os.path.join(out_path, 'DATA')
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    images_out = os.path.join(sub_folder, 'images')
    masks_out = os.path.join(sub_folder, 'masks')
    nccts_out = os.path.join(sub_folder, 'nccts')
    dwis_out = os.path.join(sub_folder, 'dwis')
    right_mask_out = os.path.join(sub_folder, 'right_hemisphere_mask')
    left_mask_out = os.path.join(sub_folder, 'left_hemisphere_mask')

    if not os.path.exists(images_out):
        os.makedirs(images_out)
    if not os.path.exists(masks_out):
        os.makedirs(masks_out)
    if not os.path.exists(nccts_out):
        os.makedirs(nccts_out)
    if not os.path.exists(dwis_out):
        os.makedirs(dwis_out)
    if not os.path.exists(right_mask_out):
        os.makedirs(right_mask_out)
    if not os.path.exists(left_mask_out):
        os.makedirs(left_mask_out)

    atlas = os.path.join(atlas_path, 'ATLAS_database')
    subjects_df = pd.read_csv(
        subjects_file,
        sep=',',
        header=None,
        names=['subject']).sort_values(by='subject').reset_index(drop=True)
    subjects_df['dl_id'] = [str(num).zfill(3) for num in range(len(subjects_df))]

    subjects = subjects_df.subject.to_list()

    subjects_df['error'] = ''  # somewhere to jot down errors

    # mediaflux folders with extra data from segmentations
    gt_1 = glob.glob(os.path.join(mediaflux_path, 'data_freda/ctp_project/CTP_DL_Data/no_seg'
                                                  '/masks_semi_ctp/*'))
    gt_2 = glob.glob(os.path.join(mediaflux_path, 'data_freda/ctp_project/CTP_DL_Data/no_seg'
                                                  '/masks_semi_ctp_june/*'))
    gt_3 = glob.glob(os.path.join(mediaflux_path,
                                  'data_freda/ctp_project/CTP_DL_Data/no_reperfusion/dwi_inference'
                                  '/segs_ctp/*'))
    ncct_gt = glob.glob(os.path.join(mediaflux_path, 'data_freda/ctp_project/CTP_DL_Data/'
                                                     'no_reperfusion/ncct_seg_reg/*'))

    for id, subject in enumerate(subjects):
        id = str(id).zfill(3)
        print(f"Running for subject: {subject}", f"id:{id}")
        if os.path.exists(
                masks_out + '/mask_' + id + '.nii.gz') and os.path.exists(
            images_out + '/image_' + id + '.nii.gz'):
            print("Already exists.")
            if not overwrite:
                continue
            else:
                print("Overwriting.")

        mistar_folder = os.path.join(atlas, subject + '/CT_baseline/CTP_baseline/mistar/')
        gt_folder = os.path.join(atlas, subject + '/CT_baseline/CTP_baseline/transform-DWI_followup/')

        dt = mistar_folder + subject + '-baseline-CTP_Delay_Time.nii.gz'
        cbf = mistar_folder + subject + '-baseline-CTP_CBF.nii.gz'
        cbv = mistar_folder + subject + '-baseline-CTP_CBV.nii.gz'
        mtt = mistar_folder + subject + '-baseline-CTP_MTT.nii.gz'


        if not os.path.exists(mistar_folder):
            mistar_folder = os.path.join(atlas, subject + '/CT_postIV/CTP_postIV/mistar/')
            gt_folder = os.path.join(atlas, subject + '/CT_postIV/CTP_postIV/transform-DWI_followup/')
            dt = mistar_folder + subject + '-postIV-CTP_Delay_Time.nii.gz'
            cbf = mistar_folder + subject + '-postIV-CTP_CBF.nii.gz'
            cbv = mistar_folder + subject + '-postIV-CTP_CBV.nii.gz'
            mtt = mistar_folder + subject + '-postIV-CTP_MTT.nii.gz'

        dt, cbf, cbv, mtt = [nb.load(path) for path in [dt, cbf, cbv, mtt]]

        # stack the four images and save
        feature_image = nb.concat_images([dt, cbf, cbv, mtt])
        nb.save(feature_image,
                images_out + '/image_' + id + '.nii.gz')

        gt = gt_folder + subject + '-space-CTP-label-DWI_manual_lesion.nii.gz'
        dwi = gt_folder + subject + '-space-CTP-DWI_b1000.nii.gz'

        if not os.path.exists(gt):
            # look into mediaflux files
            try:
                gt = [file for file in gt_1
                      if subject in file][0]
                dwi = os.path.join(mediaflux_path,
                                   'data_freda/ctp_project/CTP_DL_Data/no_seg'
                                   '/dwi_ctp/' + subject + '_dwi_ctp.nii.gz')
            except IndexError:
                try:
                    gt = [file for file in gt_2
                          if subject in file][0]
                    dwi = os.path.join(mediaflux_path,
                                       'data_freda/ctp_project/CTP_DL_Data/no_seg'
                                       '/dwi_ctp_june/' + subject + '_dwi_ctp.nii.gz')
                except IndexError:
                    try:
                        gt = [file for file in gt_3
                              if subject in file][0]
                        dwi = os.path.join(mediaflux_path,
                                           'data_freda/ctp_project/CTP_DL_Data/no_reperfusion/dwi_inference'
                                           '/dwi_ctp/' + subject + '_dwi_ctp.nii.gz')
                    except IndexError:
                        try:
                            gt = [file for file in ncct_gt
                                  if subject in file][0]
                            dwi = os.path.join(mediaflux_path,
                                               'data_freda/ctp_project/CTP_DL_Data/no_reperfusion'
                                               '/ncct_registrations/' + subject + '_ncct_reg.nii.gz')
                        except IndexError:
                            print('Could not find a ground truth for {}.'.format(subject))
                            subjects_df.loc[subjects_df.subject == subject, 'error'] = 'no gt'
                            continue

        shutil.copyfile(gt, masks_out + '/mask_' + id + '.nii.gz')
        shutil.copyfile(dwi, dwis_out + '/dwi_' + id + '.nii.gz')

        ncct_folder = mistar_folder + 'Mean_baseline/'

        if ("AU05" not in subject) and ("CN23" not in subject) and ("CN03" not in subject):
            try:
                ncct_brain = glob.glob(ncct_folder + '*baseline-mistar_brain.nii.gz')[0]
                shutil.copyfile(ncct_brain, nccts_out + '/ncct_' + id + '.nii.gz')
            except IndexError:
                try:
                    ncct_brain = glob.glob(ncct_folder + '*baseline_brain.nii.gz')[0]
                    shutil.copyfile(ncct_brain, nccts_out + '/ncct_' + id + '.nii.gz')
                except IndexError:
                    print('No skull stripped NCCT.')
                    subjects_df[subjects_df.subject == subject, 'error'] = 'no brain image'
                    continue
        else:
            try:
                ncct_brain = glob.glob(ncct_folder + '*baseline_brain.nii.gz')[0]
                shutil.copyfile(ncct_brain, nccts_out + '/ncct_' + id + '.nii.gz')
            except IndexError:
                print('No skull stripped NCCT.')
                subjects_df[subjects_df.subject == subject, 'error'] = 'no brain image'
                continue

        # generate the hemisphere masks
        infer.main(ncct_brain, left_mask_out + '/left_mask_' + id + '.nii.gz', 'left', path_to_hemisphere_masking)
        infer.main(ncct_brain, right_mask_out + '/right_mask_' + id + '.nii.gz', 'right', path_to_hemisphere_masking)

    subjects_df.to_csv(os.path.join(out_path, 'subject_ids.csv'))


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('usage: python organise_data.py subjects_file atlas_path mediaflux_path out_path overwrite=True/False')
    main(*sys.argv[1:])
