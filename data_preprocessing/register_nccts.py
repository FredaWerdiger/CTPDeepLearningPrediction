import os
import glob
import ants

'''
This is the code to run all the NCCT registrations. 
The NCCT images were selected with ncct_segmentations and the brains were extracted with DL
CL segmented the nccts
'''
def main():
    HOMEDIR = os.path.expanduser('~/')
    mediaflux = HOMEDIR + 'mediaflux/'
    atlas = HOMEDIR + 'atlas'

    if not os.path.exists(mediaflux):
        mediaflux = 'X:/'
        atlas = 'Y:/'

    out_location = os.path.join(mediaflux, 'data_freda', 'ctp_project', 'CTP_DL_Data',
                                'no_reperfusion', 'ncct_registrations')
    out_location_masks = os.path.join(mediaflux, 'data_freda', 'ctp_project', 'CTP_DL_Data',
                                'no_reperfusion', 'ncct_seg_reg')

    if not os.path.exists(out_location):
        os.makedirs(out_location)
    if not os.path.exists(out_location_masks):
        os.makedirs(out_location_masks)

    nccts = glob.glob(os.path.join(mediaflux, 'data_freda', 'ncct_segmentation', 'inference', '*.nii.gz'))
    nccts.sort()
    ncct_masks = glob.glob(
        os.path.join(mediaflux, 'data_freda', 'ncct_segmentation', 'inference', 'prediction', '*.nii.gz'))

    ncct_segs = glob.glob(os.path.join(mediaflux, 'data_freda', 'ctp_project', 'CTP_DL_Data',
                                       'no_reperfusion', 'ncct_segmentations/*'))

    subjects = [os.path.basename(name).split('_ncct')[0] for name in ncct_segs]

    rigid = ['INSP_AU010703',
             'INSP_CN030064']

    deformable = ['INSP_CN190025']

    for subject in subjects:
        if os.path.exists(os.path.join(out_location_masks, 'mask_' + subject + '.nii.gz')):
            print('{} already done'.format(subject))
            continue
        if subject in rigid:
            ttype = 'Rigid'
        elif subject in deformable:
            ttype = 'SyN'
        else:
            ttype = 'Affine'

        ncct = [name for name in nccts if subject in name][0]
        mask = [name for name in ncct_masks if subject in name][0]
        seg = [name for name in ncct_segs if subject in name][0]
        mbl = os.path.join(atlas, 'ATLAS_database', subject,
                           'CT_baseline', 'CTP_baseline', 'mistar', 'Mean_baseline',
                           subject + '-baseline-CTP_Mean_baseline_brain.nii.gz')
        mbl_mask = os.path.join(atlas, 'ATLAS_database', subject,
                                'CT_baseline', 'CTP_baseline', 'mistar', 'Mean_baseline',
                                subject + '-baseline-CTP_Mean_baseline_brainmask.nii.gz')

        if not os.path.exists(mbl_mask):
            mbl = os.path.join(atlas, 'ATLAS_database', subject,
                               'CT_baseline', 'CTP_baseline', 'mistar', 'Mean_baseline',
                               subject + '-baseline-CTP_Mean_baseline-mistar_brain.nii.gz')
            mbl_mask = os.path.join(atlas, 'ATLAS_database', subject,
                                    'CT_baseline', 'CTP_baseline', 'mistar', 'Mean_baseline',
                                    subject + '-baseline-CTP_Mean_baseline-mistar_brainmask.nii.gz')
        if not os.path.exists(mbl):
            print('can\'t find image')
            continue

        fixed = ants.image_read(mbl)
        ncct = ants.image_read(ncct)
        mask = ants.image_read(mask)
        mbl_mask = ants.image_read(mbl_mask)
        # seg = ants.image_read(seg)
        moving = ncct * mask
        print("Running transform for {}".format(subject))
        reg = ants.registration(
            fixed,
            moving,
            type_of_transform=ttype
        )
        fixedBrainReg = ants.apply_transforms(
            fixed, moving,
            reg['fwdtransforms'],
            interpolator='bSpline')
        ants.image_write(fixedBrainReg * mbl_mask, os.path.join(out_location, subject + '_ncct_reg.nii.gz'))

        # transform lesion mask
        moving = seg
        moving = ants.image_read(moving)
        fixedBrainReg = ants.apply_transforms(
            fixed, moving,
            reg['fwdtransforms'],
            interpolator='nearestNeighbor')
        ants.image_write(fixedBrainReg * mbl_mask, os.path.join(out_location_masks, 'mask_' + subject + '.nii.gz'))


if __name__ == '__main__':
    main()
