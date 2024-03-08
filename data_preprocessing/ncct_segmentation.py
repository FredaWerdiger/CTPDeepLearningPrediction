import os
import pandas as pd
import SimpleITK as sitk
import shutil
import glob

'''
This code records how the NCCT images were extracted for the project
'''
HOMEDIR = os.path.expanduser('~/')

atlas = HOMEDIR + 'atlas'
mediaflux = HOMEDIR + 'mediaflux'

if not os.path.exists(atlas):
    atlas = 'Y:/'
    mediaflux = 'X:/'

no_reperfusion= pd.read_csv(mediaflux + '/data_freda/ctp_project/CTP_DL_Data/no_reperfusion/no_reperfusion_all.csv')

nccts = no_reperfusion[no_reperfusion.apply(lambda x: (x.note == 'follow_up_ncct') and (x.slab != 1.0), axis=1)]
out_dir = os.path.join(mediaflux, 'data_freda', 'ncct_segmentation', 'inference')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for ind in nccts.index:
    line = nccts.loc[ind]
    subject = line.subject
    print(subject)
    if type(line.ncct_fu_day) == str:
        ncct_dir = glob.glob(os.path.join(atlas, 'ATLAS_database', subject, 'CT-follow_up_' + line.ncct_fu_day, 'NCCT*'))[0]
        tilts = glob.glob(os.path.join(ncct_dir, '*Tilt*'))
        if len(tilts) == 1:
            ncct = tilts[0]
        elif len(tilts) == 2:
            ncct = [tilt for tilt in tilts if "thick" in tilt]
            if ncct:
                ncct = ncct[0]
            else:
                continue  # to difficult
        else:
            ncct = os.path.join(ncct_dir, subject + '-fu-NCCT.nii.gz')
            if not os.path.exists(ncct):
                ncct = os.path.join(atlas, 'ATLAS_database', subject, 'CT-follow_up',
                                    'NCCT-follow_up', subject + '-fu-NCCT_thick.nii.gz')
    else:
        ncct_dir = glob.glob(
            os.path.join(atlas, 'ATLAS_database', subject, 'CT-follow_up', 'NCCT*'))
        if not ncct_dir:
            continue
        ncct_dir = ncct_dir[0]
        tilts = glob.glob(os.path.join(ncct_dir, '*Tilt*'))
        if len(tilts) == 1:
            ncct = tilts[0]
        elif len(tilts) == 2:
            ncct = [tilt for tilt in tilts if "thick" in tilt]
            if ncct:
                ncct = ncct[0]
            else:
                continue  # to difficult
        else:
            ncct = os.path.join(ncct_dir, subject + '-fu-NCCT.nii.gz')
            if not os.path.exists(ncct):
                ncct = os.path.join(atlas, 'ATLAS_database', subject, 'CT-follow_up',
                                    'NCCT-follow_up', subject + '-fu-NCCT_thick.nii.gz')
    if os.path.exists(ncct):
        im = sitk.ReadImage(ncct)
        if len(im.GetSize()) != 3:
            shutil.copyfile(ncct, os.path.join(out_dir, subject + '_ncct.nii.gz'))
