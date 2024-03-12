# CTP Deep Learning 
Codes to train a Deep Learning Model to predict infarct in stroke patients.
## organise_data.py
Data is organised into the Deep Learning format expected by the training code i.e. images/ and masks/
Other data is also extracted:
- Mean Baseline Images are organised into ncct folder, for training.
- Left and Right hemisphere masks are extracted for statistical analysis of results. This utilises the inferference code (infer.py) found in the github.com/FredaWerdiger/automatic_rotation repository. These are not necessary to running the code.
- DWIs - the registered DWIs are saved into /dwis/ folder (or the follow-up nccts if that was the case). These are used only to generate results png images and are not necessary to running the codes.

### usage
python organise_data.py subjects_file atlas_path mediaflux_path outpath overwrite
#### inputs
- subjects_file - a csv file with the INSPIRE ID of all patients you wish to include. An example is included here, subjects_file.csv
- atlas_path - the path to the loaded ATLAS mediaflux drive, i.e., X:/
- mediaflux_path - the path to the loaded mediaflux with the project data (where /data_freda/ is located), i.e., Y:/
- outpath - Where you want the data to output. If you name a folder, e.g. /myfolder/ the data will appear as /myfolder/DATA/
- overwrite - True if you want to rerun everything, False if you want to just be able to start and stop the processing without having to write over everything.
#### outputs
Aside from all the data, which will be saved as /outpath/DATA, a csv file will be saved called subject_ids.csv with the INSPIRE IDs and corresponding Deep Learning IDs, and notice of any cases that did not successfully run under an 'error' column.

  ## train.py
  Training the actual Deep Learning model. The code:
  - Loads the data from DATA/
  - Stratifies into train/validation/test
  - trains the model and saves the best model
  - Saves a loss plot
  - Saves a text file with results and information on the model for reproduction
  - Saves the predictions as well and png files to summarise results.
  - Saves a results csv with all statistics on performance as well as lesion size for the test set.
### usage
python train.py data_dir out_tag test_cases features image_size max_epochs notes
#### input
- data_dir - the only required input. The location of the '../DATA' created by organise_data.py (path should end in 'DATA'
- out_tag -  This is the postfix for the folder that will contain results. It will appear as '../DATA/out_'out_tag'/. Default is '', i.e., '../DATA/out_/'
- test_cases - a list of cases that will be sent to the testing dataset - i.e., challenging cases. These have to be entered as the Deep Learning IDs that correspond with the ../DATA/images/image_<dl_dl>.nii.gz files. An example is included in the test_cases_dl_id.csv file.
- features - which features you want to train on. Default is ['DT', 'CBF', 'CBV', 'ncct'].
- image_size - the size of the images for training. Default is 128.
- max_epoch - number of epochs for training. Default is 400.
- notes - any additional comments to add to the results text file for your own reference.

#### output
- saved model
- 'model_info....txt' - text file recording key information. 
- 'loss_plot....png' - Plot of train and validation metrics to ensure no overfitting.
- binary predictions as niftis into /DATA/out_<out_tag>/pred
- probability maps as niftis into /DATA/out_<out_tag>/proba
- predictions as pngs - probability maps /DATA/out_<out_tag>/pngs
- auc plot which will only contain three points as per binary outputs.
## Data preprocessing
This contains all the codes used in the preprocessing of the data. These are included so that data selection can be reproduced. 
  

## Dependencies
- MONAI version: 0.10.dev2237
- Numpy version: 1.21.2
- Pytorch version: 1.10.2
