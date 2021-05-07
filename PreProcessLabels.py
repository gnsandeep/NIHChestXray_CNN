import pandas as pd
from shutil import copyfile

"""
This needs to run only once to format the labels file to a format that can be loaded in Pytorch dataset
"""




"""
define the 14 labels 
"""
labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'No Finding',
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']

"""
Read the orginial labels file , drop unwanted columns and convert into a format 
that can be used for training/validation in PyTorch
Original Format:
00000001_001.png,Cardiomegaly|Emphysema,1,1,58,M,PA,2894,2729,0.14300000000000002,0.14300000000000002
Format for Training:
1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,00000001_001.png,1
"""
org_df = pd.read_csv('Data_Entry_2017_v2020.csv')
def update_toMClass(row):
    label_dict = {v:0 for v in labels}
    imgname = row['Image Index']
    label_dict['Image Index'] = row['Image Index']
    label_dict['Patient ID'] = row['Patient ID']
    label_lsit = row['Finding Labels'].split('|')
    for label in label_lsit:
        label_dict[label] = 1
        
    return pd.Series(label_dict)



"""
Save updated format to file for future use
"""
org_df_updated = org_df.apply(update_toMClass,axis=1)
org_df_updated.to_csv("Updated_Data_Entry_2017.csv", index=False)


