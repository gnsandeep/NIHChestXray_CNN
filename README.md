# NIHChestXray_CNN

## Background
National Institute of Health open sourced 112,000+ images of chest chest x-rays. This dataset was opened in order to allow clinicians to make better diagnostic decisions for patients with various lung diseases. This project reproduces model training, predictions, and heatmaps from https://nihcc.app.box.com/v/ChestXray-NIHCC/file/256057377774 paper that predicted 14 common diagnoses using convolutional neural networks for National Institute of Health Chest X-ray Images. <br/>This Project is part of CS598 Deep Learning for Healthcare graduate course at UIUC.

## NIH Dataset
Go to https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737  or use batch_download_zips.py to download all the image archive files and extract all tar.gz files to a single folder, and update path in load() function in notebooks or python files. The Class labels are provided in Data_Entry_2017_v2020.csv , which is available at https://nihcc.app.box.com/v/ChestXray-NIHCC. Data_Entry_2017_v2020.csv is also available in this repository.

## Environment
Conda environment is recommended. use provided environment.yml to create conda environment and run the models.

## Preprocessing
PreProcessLabels.py reads the originial labels file (Data_Entry_2017_v2020.csv) , drops unwanted columns and convert into a format (Updated_Data_Entry_2017.csv) that can be used for training/validation in PyTorch

Original Format <br/>
Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y]<br/> 
00000001_001.png,Cardiomegaly|Emphysema,1,1,58,M,PA,2894,2729,0.14300000000000002,0.14300000000000002 

Updated Format <br/>
Cardiomegaly,Emphysema,Effusion,Hernia,Infiltration,Mass,Nodule,No Finding,Atelectasis,Pneumothorax,Pleural_Thickening,Pneumonia,Fibrosis,Edema,Consolidation,Image Index,Patient ID 
1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,00000001_001.png,1

CreateTrainValData.py splits the data set in to Training and validation. train_1.csv and val_1.csv is for full dataset. For evaluation purpose we can set the sample size as described in CreateTrainValData.py and then generate Train and Validation Data. train_



## Training
GPU is required to train the model on full dataset. Please refer Experimental Evaluation Section if you are planning to train and explore on a small dataset. Resnet50 and DenseNet121 are trained on Full Dataset. train_0.1.csv and val-small_0.1.csv is for 10% sample. train_ and val_ is for 1% sample.

Running ResNet50FD.py and DenseNet121FD.py will train CNN models based on ResNet and DenseNet. Trained Model and results are stored in ResNet and DenseNet Folders.

Example : python -u ResNet50FD.py > resnetout.txt &.  <br/>
Note : It takes 10 hours on AWS p2.xlarge ( 1 GPU )

Training is done for 10 Epoch for Full dataset. It was trained on AWS Deep Learning VM p2.xlarge ( which has one GPU) . It took 10 hours for 10 epochs. Training , Validation loss and AUC values starts to converge after 7 epochs. We will be using AUC ROC for evaluation. 

AUC values for these models are visualized in Visualization.ipynb . Visualization of disease localization using Class Activation Mapping is also provided in Visualization.ipynb for the trained model.

## Results


## Evaluation 
DenseNet and ResNet models were trained and evaluated on smaller dataset to fine tune hyper parameters like learning rate , weight decay , optimizer. A Simple CNN model was also used.

SimpleCNN.ipynb , ResNet50.ipynb  DenseNet121.ipynb was trained on 10% dataset. These models used base SGD as optimizer.
SimpleCNNSGDMomentum.ipynb , ResNet50SGDMomentum.ipynb ,  DenseNet121SGDMomentum.ipynb was trained on 10% dataset. These models used base SGD wiht momentum as optimizer.

AUC values for these models are visualized in Visualization.ipynb
