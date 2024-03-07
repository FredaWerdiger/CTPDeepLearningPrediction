import os
import pandas as pd
import SimpleITK as sitk
import sys
import glob

'''
This records all the images used in the CTP project for no reperfusion patients
last run in March 2024
'''

df_no = pd.read_excel('INSPIRE no reperfusion Pts Freda_2023-080-27.xlsx',
                      sheet_name=[0],
                      header=[0])[0]

df_partial = pd.read_excel('INSPIRE no reperfusion Pts Freda_2023-080-27.xlsx',
                           sheet_name=[1],
                           header=[0])[1]

df_no_complete = df_no[df_no.apply(
    lambda x: (x['Occlusion severity (TIMI:0=complete occlusion, 3=normal)'] in [0, 1])
              and (str(x[
                           'Occlusion location (no visible occlusion=0,ica=1, m1=2, m2=3, m3=4, aca=5, pca=6, basilar=7, vertebral=8)']) != '6'),
    axis=1)]
df_partial_complete = df_partial[df_partial.apply(
    lambda x: (x['Occlusion severity (TIMI:0=complete occlusion, 3=normal)'] in [0, 1])
              and (str(x[
                           'Occlusion location (no visible occlusion=0,ica=1, m1=2, m2=3, m3=4, aca=5, pca=6, basilar=7, vertebral=8)']) != '6'),
    axis=1
)]

HOMEDIR = os.path.expanduser('~/')
atlas = 'Y:/'
mediaflux = 'X:/'

if os.path.exists(HOMEDIR + 'mediaflux'):
    mediaflux = HOMEDIR + 'mediaflux/'
    atlas = HOMEDIR + 'atlas/'

mediaflux_subjects = [name for name in os.listdir(atlas + 'ATLAS_database') if 'INSP_' in name]
missing_subjects = []

for subject in df_partial_complete['INSPIRE ID'].to_list():
    if subject not in mediaflux_subjects:
        missing_subjects.append(subject)

for subject in df_no_complete['INSPIRE ID'].to_list():
    if subject not in mediaflux_subjects:
        missing_subjects.append(subject)

with open('./study_lists/missing_subjects_no_reperfusion.txt', 'w') as myfile:
    for subject in missing_subjects:
        myfile.write(subject)
        myfile.write(',')

all_subjects = [subject for subject in
                df_partial_complete['INSPIRE ID'].to_list() + df_no_complete['INSPIRE ID'].to_list()
                if subject not in missing_subjects]

missing_ctp = []
missing_dwi = []
has_all_imaging = []
has_one_type = []
follow_up_ncct = []
has_neither = []
post_iv = []
has_manual_segmentation = []
no_adc = []

ncct_pngs_outdir = mediaflux + 'data_freda/ctp_project/CTP_DL_Data/no_reperfusion/ncct_pngs/'
mr_pngs_outdir = mediaflux + 'data_freda/ctp_project/CTP_DL_Data/no_reperfusion/mr_pngs/'

if not os.path.exists(ncct_pngs_outdir):
    os.makedirs(ncct_pngs_outdir)
if not os.path.exists(mr_pngs_outdir):
    os.makedirs(mr_pngs_outdir)

for subject in all_subjects:
    a = 0
    print(subject)
    if os.path.exists(os.path.join(atlas, 'ATLAS_database', subject, 'CT_baseline', 'CTP_baseline')):
        print('Has CTP imaging')
        a += 1
    elif os.path.exists(os.path.join(atlas, 'ATLAS_database', subject, 'CT_postIV', 'CTP_postIV')):
        print('Has CTP post IV')
        post_iv.append(subject)
    else:
        missing_ctp.append(subject)
    if glob.glob(os.path.join(atlas, 'ATLAS_database', subject, 'MR-follow_up*', 'DWI-follow_up*')):
        print('has DWI imaging')
        a += 1
        seg = glob.glob(os.path.join(atlas, 'ATLAS_database', subject, 'MR-follow_up*', 'DWI-follow_up*', '*lesion*'))
        if seg:
            print('has manual segmentation')
            has_manual_segmentation.append(subject)
        else:
            if not os.path.exists(os.path.join(atlas, 'ATLAS_database', subject, 'MR-follow_up', 'ADC-follow_up')):
                print('no adc')
                no_adc.append(subject)

    elif glob.glob(os.path.join(atlas, 'ATLAS_database', subject, 'CT-follow_up*', 'NCCT-follow_up*')):
        print('has follow-up NCCT')
        follow_up_ncct.append(subject)
        ncct = os.path.join(atlas, 'ATLAS_database', subject, 'CT-follow_up',
                            'NCCT-follow_up', subject + '-fu-NCCT.nii.gz')
        if not os.path.exists(ncct):
            ncct = os.path.join(atlas, 'ATLAS_database', subject, 'CT-follow_up',
                                'NCCT-follow_up', subject + '-fu-NCCT_thick.nii.gz')
        if not os.path.exists(ncct):
            for path in glob.glob(os.path.join(atlas, 'ATLAS_database', subject, 'CT-follow_up*', 'NCCT-follow_up*')):
                tilts = glob.glob(os.path.join(path, '*Tilt*'))
                if len(tilts) == 1:
                    ncct = tilts[0]
                elif len(tilts) == 2:
                    ncct = [tilt for tilt in tilts if "thick" in tilt]
                    if ncct:
                        ncct = ncct[0]
                    else:
                        continue  # to difficult
                else:
                    ncct = os.path.join(path, subject + '-fu-NCCT.nii.gz')
                    if not os.path.exists(ncct):
                        ncct = os.path.join(atlas, 'ATLAS_database', subject, 'CT-follow_up',
                                            'NCCT-follow_up', subject + '-fu-NCCT_thick.nii.gz')
                # get the time stamp
                path_components = path.split(os.sep)
                if path_components[-1].split('_')[-1] == 'up':
                    time = path_components[-2].split('_')[-1]
                else:
                    time = path_components[-1].split('_')[-1]

    else:
        missing_dwi.append(subject)
    if a == 0:
        print('Has none imaging')
        has_neither.append(subject)
    elif a == 2:
        print('Has all imaging')
        has_all_imaging.append(subject)
    elif a == 1:
        has_one_type.append(subject)

post_iv = [patient for patient in post_iv if patient in has_one_type]

# with open('./study_lists/no_images_no_reperfusion.txt', 'w') as myfile:
#     for subject in has_neither:
#         myfile.write(subject)
#         myfile.write(',')

# with open('./study_lists/missing_dwi_no_reperfusion.txt', 'w') as myfile:
#     for subject in missing_dwi:
#         myfile.write(subject)
#         myfile.write(',')


# checking NCCT

exclude = [
    'INSP_AU010066',  # not much there, check with MV
    'INSP_AU100189',  # huge bleed
    'INSP_CN030032',  # mass effect
    'INSP_CN030157',  # blood and artefact
    'INSP_CN030296',  # mass effect
    'INSP_AU010335',  # bleed
    'INSP_AU010507',  # bleed
    'INSP_AU240020',  # bleed
    'INSP_CN090060',  # bleed
    'INSP_CN090071',
    'INSP_CN130047',
    'INSP_AU240073',  # bleed
    'INSP_CN120007',  # mass effect
    'INSP_CN190019'  # bleed

]
difficult = [
    'INSP_AU050097',
    'INSP_AU100682',
    'INSP_AU240109',
    'INSP_CA020081',
    'INSP_CN030249',
    'INSP_AU020116',
    'INSP_CN190019'  # bleeding and mass effect
    'INSP_AU240073',  # bleeding
]

manual_segmentation = [
    'INSP_AU010703',
    'INSP_AU010747',
    'INSP_AU010811',
    'INSP_AU100087',
    'INSP_AU160010',
    'INSP_AU240050',  # bleed
    'INSP_AU240051',
    'INSP_AU240071',
    'INSP_AU240081',
    'INSP_CN020284',
    'INSP_CN020331',
    'INSP_CN030035',
    'INSP_CN030059',
    'INSP_CN030064',
    'INSP_CN030073',
    'INSP_CN030079',
    'INSP_CN030126',
    'INSP_CN090005',
    'INSP_CN090024',  ##last one i did
    'INSP_AU010581',
    'INSP_AU010356',
    'INSP_AU230139',
    'INSP_AU010699',
    'INSP_AU010589',
    'INSP_AU010386',
    'INSP_AU010341',  # bleed
    'INSP_CN090049',
    'INSP_CN090059',
    'INSP_CN090079',
    'INSP_CN120003',
    'INSP_CN130027',
    'INSP_CN130081',
    'INSP_CN150013',
    'INSP_CN150088',
    'INSP_CN030033',  # USE  _5d
    'INSP_CN190025',  # use _2d
    'INSP_CN130082',  # use _4d (there is only one)
    'INSP_CN190011',  # use _2d
    'INSP_CA010042',  # use 1d
]

follow_up_ncct = [sub for sub in follow_up_ncct if sub not in exclude]

# with open('./study_lists/follow_up_ncct_no_reperfusion.txt', 'w') as myfile:
#     for subject in follow_up_ncct:
#         myfile.write(subject)
#         myfile.write(',')


df_full_imaging = pd.DataFrame(has_all_imaging, columns=['subject'])
df_full_imaging['note'] = 'full_imaging'

df_postiv = pd.DataFrame(post_iv, columns=['subject'])
df_postiv['note'] = 'post_iv_ctp'

df_ncct = pd.DataFrame(follow_up_ncct, columns=['subject'])
df_ncct['note'] = 'follow_up_ncct'

df_no_reperfusion = pd.concat([df_full_imaging, df_ncct, df_postiv])

# how many patients already have segmentations?

to_infer = [sub for sub in df_no_reperfusion.subject.to_list() if sub not in has_manual_segmentation]

df_no_reperfusion['segmentation_type'] = df_no_reperfusion['subject'].apply(
    lambda x: 'infer' if x in to_infer else 'manual')

for subject in no_adc:
    df_no_reperfusion.loc[df_no_reperfusion.subject == subject, 'segmentation_type'] = 'infer_no_adc'

df_no_reperfusion['ncct_fu_day'] = ''

df_no_reperfusion.loc[df_no_reperfusion.subject == 'INSP_CN030033', 'ncct_fu_day'] = '5d'
df_no_reperfusion.loc[df_no_reperfusion.subject == 'INSP_CN190025', 'ncct_fu_day'] = '2d'
df_no_reperfusion.loc[df_no_reperfusion.subject == 'INSP_CN130082', 'ncct_fu_day'] = '4d'
df_no_reperfusion.loc[df_no_reperfusion.subject == 'INSP_CN190011', 'ncct_fu_day'] = '2d'
df_no_reperfusion.loc[df_no_reperfusion.subject == 'INSP_CA010042', 'ncct_fu_day'] = '1d'

dwi_subs = df_no_reperfusion[df_no_reperfusion.apply(
    lambda x: x.note in ['full_imaging', 'post_iv_ctp'], axis=1)].subject.to_list()

has_transform = []
for subject in dwi_subs:

    df_full_imaging.to_csv('./study_lists/full_data_no_reperfusion.csv', index=False)
    if df_no_reperfusion.loc[df_no_reperfusion.subject == subject, 'note'].values[0] == 'full_imaging':
        path = os.path.join(atlas, 'ATLAS_database', subject,
                            'CT_baseline',
                            'CTP_baseline',
                            'transform-DWI_followup')
    elif df_no_reperfusion.loc[df_no_reperfusion.subject == subject, 'note'].values[0] == 'post_iv_ctp':
        path = os.path.join(atlas, 'ATLAS_database', subject,
                            'CT_postIV', 'CTP_postIV',
                            'transform-DWI_followup')
    if os.path.exists(path):
        has_transform.append(subject)

no_transform = [sub for sub in dwi_subs if sub not in has_transform]

# reasons for no transform?
slabs = []
not_processed = []
df_no_reperfusion['slab'] = ''
for subject in df_no_reperfusion.subject.to_list():
    if df_no_reperfusion.loc[df_no_reperfusion.subject == subject, 'note'].values[0] in ['full_imaging',
                                                                                         'follow_up_ncct']:
        path = os.path.join(atlas, 'ATLAS_database', subject,
                            'CT_baseline',
                            'CTP_baseline',
                            'mistar')
    elif df_no_reperfusion.loc[df_no_reperfusion.subject == subject, 'note'].values[0] == 'post_iv_ctp':
        path = os.path.join(atlas, 'ATLAS_database', subject,
                            'CT_postIV', 'CTP_postIV',
                            'mistar')
    if os.path.exists(path):
        if any('slab' in name.lower() for name in os.listdir(path)):
            slabs.append(subject)
            df_no_reperfusion.loc[df_no_reperfusion.subject == subject, 'slab'] = 1
    else:
        not_processed.append(subject)

no_transform = [sub for sub in no_transform if (sub not in slabs) and (sub not in not_processed)]

# following subjects are not processed

not_processed = [
    'INSP_AU010043',  # Unable to detect VOF
    'INSP_AU100070',  # DICOM error
    'INSP_AU100129',
    'INSP_CA010042'
    # Dynamic scan time too short
]

# 'INSP_AU100025', #not sure the status of these cases

for subject in not_processed:
    index = df_no_reperfusion.loc[df_no_reperfusion.subject == subject].index.values[0]
    df_no_reperfusion.drop(index=index, inplace=True)

df_no_reperfusion.sort_values(by='subject', inplace=True)
df_no_reperfusion.reset_index(drop=True, inplace=True)

## QC exclusions
exclude_qc = [
    'INSP_AU010017', 'INSP_AU010278', 'INSP_AU010281',
    'INSP_AU010352', 'INSP_AU010409', 'INSP_AU010413',
    'INSP_AU010427', 'INSP_AU010433', 'INSP_AU010460',
    'INSP_AU010461', 'INSP_AU010470', 'INSP_AU010479',
    'INSP_AU010494', 'INSP_AU010547', 'INSP_AU010655',
    'INSP_AU010691', 'INSP_AU010698', 'INSP_AU010712',
    'INSP_AU010717', 'INSP_AU010734', 'INSP_AU010738',
    'INSP_AU010756', 'INSP_AU010771', 'INSP_AU010788',
    'INSP_AU010829', 'INSP_AU010830', 'INSP_AU010840',
    'INSP_AU010852', 'INSP_AU010873', 'INSP_AU010904',
    'INSP_AU010916',
    'INSP_AU020079',  # normal
    'INSP_AU100043',  # no follow-up infarct
    'INSP_AU100065',  # no core growth
    'INSP_AU100151',  # no growth
    'INSP_AU100295',  # no perfusion lesion
    'INSP_AU100311',  # no follow up infarct
    'INSP_AU100656',  # POCI
    'INSP_AU100721',  # no Follow up infarct
    'INSP_AU170008',  # no follow up infarct
    'INSP_AU230031',  # bad perfusion imaging
    'INSP_AU240084',  # no follow up infarct
    'INSP_CA020089',  # no fu infarct
    'INSP_CN020060',  # bad imaging
    'INSP_CN020178',  # brain stem, no perfusion lesion
    'INSP_CN020403', 'INSP_AU040019', 'INSP_AU050085',
    'INSP_AU100185', 'INSP_AU100432', 'INSP_AU100721',
    'INSP_CN020024', 'INSP_CN020081', 'INSP_CN020339',
    'INSP_CN020361', 'INSP_CN030040', 'INSP_CN030062',
    'INSP_CN030098', 'INSP_CN130070', 'INSP_CN130073',
    'INSP_CN190020', 'INSP_AU020075', 'INSP_AU020150',
    'INSP_AU100141', 'INSP_CN020199', 'INSP_CN020414',
    'INSP_CN030039', 'INSP_CN030049', 'INSP_CN130040',
    'INSP_CN030106', 'INSP_AU020116', 'INSP_AU050097',
    'INSP_AU100087', 'INSP_AU100682', 'INSP_AU160010',
    'INSP_AU240050',
    'INSP_AU240051',
    'INSP_AU240081',
    'INSP_AU240109',
    'INSP_CA020081',
    'INSP_CN020284',
    'INSP_CN030079',
    'INSP_CN090024',
    'INSP_CN090059',
    'INSP_CN120003',
    'INSP_CN150013',
    'INSP_CN020331',
    'INSP_AU100201',
    'INSP_AU100209',
    'INSP_CN020372',
    'INSP_CN020413',
    'INSP_AU100018',
    'INSP_CN020268',
    'INSP_CN020372',
    'INSP_CN020283',
    'INSP_CN020321' ,# too much bleeding
    'INSP_AU100025' # old infarct with perfusion deficit
]

exclude_atlas= ['INSP_AU100024', # subjects added to atlas later on that are no in the study
                 'INSP_AU100029',
                 'INSP_AU100052',
                 'INSP_AU100054',
                 'INSP_AU100057',
                 'INSP_AU100060',
                 'INSP_AU100074',
                 'INSP_AU100076',
                 'INSP_AU100077']
# exclude_check_df = df_no_reperfusion[df_no_reperfusion.apply(lambda x: x.subject in exclude_check, axis=1)]
# include_check_df = df_no_reperfusion[df_no_reperfusion.apply(lambda x: x.subject in include_check, axis=1)]
#
# exclude_check_df.to_csv('./study_lists/no_reperfusion_check_to_exclude.csv', index=None)
# include_check_df.to_csv('./study_lists/no_reperfusion_check_to_include.csv', index=None)

for subject in exclude_qc + exclude_atlas:
    try:
        index = df_no_reperfusion.loc[df_no_reperfusion.subject == subject].index.values[0]
        df_no_reperfusion.drop(index=index, inplace=True)
    except IndexError:
        continue  # its not there anymore

# removing slabs
df_no_reperfusion = df_no_reperfusion[df_no_reperfusion.slab != 1]

clinical_df = df_no_reperfusion.join(
    df_partial_complete.set_index('INSPIRE ID', drop=True),
    on='subject',
    how='left')

include_checked = ['INSP_AU010025', 'INSP_AU010040', 'INSP_AU010140',
                   'INSP_AU010151', 'INSP_AU010261', 'INSP_AU010291',
                   'INSP_AU010408', 'INSP_AU010414', 'INSP_AU010421',
                   'INSP_AU010422', 'INSP_AU010439', 'INSP_AU010450',
                   'INSP_AU010454', 'INSP_AU010476', 'INSP_AU010506',
                   'INSP_AU010516', 'INSP_AU010569', 'INSP_AU010592',
                   'INSP_AU010611', 'INSP_AU010612', 'INSP_AU010622',
                   'INSP_AU010626', 'INSP_AU010658', 'INSP_AU010672',
                   'INSP_AU010690', 'INSP_AU010708', 'INSP_AU010714',
                   'INSP_AU010715', 'INSP_AU010716', 'INSP_AU010720',
                   'INSP_AU010729', 'INSP_AU010742', 'INSP_AU010757',
                   'INSP_AU010759', 'INSP_AU010775', 'INSP_AU010777',
                   'INSP_AU010778', 'INSP_AU010780', 'INSP_AU010782',
                   'INSP_AU010806', 'INSP_AU010815', 'INSP_AU010816',
                   'INSP_AU010819', 'INSP_AU010827', 'INSP_AU010832',
                   'INSP_AU010834', 'INSP_AU010850', 'INSP_AU010922',
                   'INSP_AU010934', 'INSP_AU010968', 'INSP_AU011011',
                   'INSP_AU011012', 'INSP_AU040021', 'INSP_AU040027',
                   'INSP_AU090006', 'INSP_AU090025', 'INSP_AU100114',
                   'INSP_AU100214', 'INSP_AU100238', 'INSP_AU100297',
                   'INSP_AU100307', 'INSP_AU100359', 'INSP_AU100491',
                   'INSP_AU100730', 'INSP_AU100739', 'INSP_AU100793',
                   'INSP_AU160016', 'INSP_AU230009', 'INSP_AU230059',
                   'INSP_AU230110', 'INSP_CN020032', 'INSP_CN020169',
                   'INSP_CN020289', 'INSP_CN020301', 'INSP_CN020391',
                   'INSP_CN030005', 'INSP_CN030027', 'INSP_CN030041',
                   'INSP_CN030048', 'INSP_CN030078', 'INSP_CN030080',
                   'INSP_CN030096', 'INSP_CN030101', 'INSP_CN030103',
                   'INSP_CN030116', 'INSP_CN030127', 'INSP_CN030129',
                   'INSP_CN120006', 'INSP_CN150030', 'INSP_CN160002',
                   'INSP_AU100034', 'INSP_AU100072', 'INSP_AU100133',
                   'INSP_AU100255', 'INSP_CN030068', 'INSP_AU010699',
                   'INSP_AU010747', 'INSP_AU010811', 'INSP_CN030033',
                   'INSP_CN030035', 'INSP_CN030059', 'INSP_CN030064',
                   'INSP_CN030073', 'INSP_CN090005', 'INSP_CN090049',
                   'INSP_CN090079', 'INSP_CN130027', 'INSP_CN130081',
                   'INSP_CN130082', 'INSP_CN190025', 'INSP_CN150088',
                   'INSP_CN190011', 'INSP_AU100025', 'INSP_AU230139',
                   'INSP_AU010703', 'INSP_CN020321', 'INSP_CN030112',
                   'INSP_AU100709', 'INSP_AU011026', 'INSP_AU010437',
                   'INSP_AU100218'
                   ]

# the case that has dwi imaging even though it says it doesnt
df_no_reperfusion.loc[df_no_reperfusion.subject == 'INSP_AU010747', 'note'] = 'full_imaging'
df_no_reperfusion.loc[df_no_reperfusion.subject == 'INSP_AU010747', 'segmentation_type'] = 'manual'

df_no_reperfusion['include_checked'] = df_no_reperfusion.apply(lambda x: 1 if x.subject in include_checked else 0,
                                                               axis=1)
df_no_reperfusion.sort_values(by=['include_checked', 'note', 'subject'], ascending=True, inplace=True)

# Below are the "cautious" patients to be place in the test set
test_subjects = [
    'INSP_CN030064',
    'INSP_CN030064',
    'INSP_AU010040',
    'INSP_AU010592',
    'INSP_AU010622',
    'INSP_AU010968',
    'INSP_AU040027',
    'INSP_AU100214',
    'INSP_AU100307',
    'INSP_AU230009',
    'INSP_AU230059',
    'INSP_CN020391',
    'INSP_CN030041',
    'INSP_CN030068',
    'INSP_CN120006'
]
df_no_reperfusion['test_cases'] = ''

for subject in test_subjects:
    df_no_reperfusion.loc[df_no_reperfusion.subject == 'INSP_AU010747', 'test_cases'] = 1
df_no_reperfusion.to_csv('./study_lists/no_reperfusion_all.csv', index=False)
