# NIHChestXray_CNN

## Background
National Institute of Health open sourced 112,000+ images of chest chest x-rays. This dataset was opened in order to allow clinicians to make better diagnostic decisions for patients with various lung diseases. This project reproduces model training, predictions, and heatmaps from https://nihcc.app.box.com/v/ChestXray-NIHCC/file/256057377774 paper that predicted 14 common diagnoses using convolutional neural networks for National Institute of Health Chest X-ray Images. <br/>This Project is part of CS598 Deep Learning for Healthcare graduate course at UIUC.

## NIH Dataset
Go to https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737  or use batch_download_zips.py to download all the image archive files and extract all tar.gz files to a single folder, and update path in load() function in notebooks or python files. The Class labels are provided in Data_Entry_2017_v2020.csv , which is available at https://nihcc.app.box.com/v/ChestXray-NIHCC. Data_Entry_2017_v2020.csv is also available in this repository.

## Sample Data
Sample Train and Validation data set in under data folder. Training tar file is split into multiple files. train_small_updated.csv and val_small_updated.csv is for 1% sample and we have images for these labels in data folder. train_0.1.csv and val-small_0.1.csv is for 10% sample . train_1.csv and val_1.csv is for full data set. 1% sample data is under data folder, please update load() function to use the correct data folder.

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

CreateTrainValData.py splits the data set in to Training and validation. train_1.csv and val_1.csv is for full dataset. For evaluation purpose we can set the sample size as described in CreateTrainValData.py and then generate Train and Validation Data. 



## Training
GPU is required to train the model on full dataset. Please refer Experimental Evaluation Section if you are planning to train and explore on a small dataset. Resnet50 and DenseNet121 are trained on Full Dataset. 

Running ResNet50FD.py and DenseNet121FD.py will train CNN models based on ResNet and DenseNet. Trained Model and results are stored in ResNet and DenseNet Folders.

Example : python -u ResNet50FD.py > resnetout.txt &.  <br/>
Note : It takes 10 hours on AWS p2.xlarge ( 1 GPU )

Training is done for 10 Epoch for Full dataset. It was trained on AWS Deep Learning VM p2.xlarge ( which has one GPU) . It took 10 hours for 10 epochs. Training , Validation loss and AUC values starts to converge after 7 epochs. We will be using AUC ROC for evaluation. 

AUC values for these models are visualized in Visualization.ipynb . Visualization of disease localization using Class Activation Mapping is also provided in Visualization.ipynb for the trained model.

## Results

| Category  | NIH Paper | DenseNet121 | ResNet50 |
| ------------- | ------------- | ------------- | ------------- |
| Atelectasis  | 0.7003  | 0.8145| 0.8048 | 
| Atelectasis	| 0.7003	| 0.8145 |	0.8048 |
| Cardiomegaly | 0.8100 | 0.9054 | 0.8992 |
| Consolidation | 0.7032 | 0.8117 | 0.8034 |
| Edema | 0.8052 | 0.9062 | 0.8977 |
| Effusion | 0.7585 | 0.8828 | 0.8807 |
| Emphysema | 0.8330 | 0.9206 | 0.9108 |
| Fibrosis | 0.7859 | 0.8075 | 0.7990 |
| Hernia | 0.8717 | 0.8922 | 0.8242 |
| Infiltration | 0.6614 | 0.7195 | 0.7080 |
| Mass | 0.6933 | 0.8675 | 0.8575 |
| Nodule | 0.6687 | 0.7693 | 0.7558 |
| Pleural Thickening | 0.6835 | 0.8018 | 0.7917 |
| Pneumonia | 0.6580 | 0.7616 | 0.7410 |
| Pneumothorax | 0.7993 | 0.8892 | 0.8859 |



## Evaluation 
DenseNet and ResNet models were trained and evaluated on smaller dataset to fine tune hyper parameters like learning rate , weight decay , optimizer. A Simple CNN model was also used.

SimpleCNN.ipynb , ResNet50.ipynb  DenseNet121.ipynb was trained on 10% dataset. These models used base SGD as optimizer.
SimpleCNNSGDMomentum.ipynb , ResNet50SGDMomentum.ipynb ,  DenseNet121SGDMomentum.ipynb was trained on 10% dataset. These models used base SGD with momentum as optimizer.

AUC values for these models are visualized in Visualization.ipynb
