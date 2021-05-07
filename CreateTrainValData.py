import pandas as pd 
from shutil import copyfile
from sklearn.model_selection import train_test_split

"""
Use this file to split data into training and validation.
How to use this file:
Sample size range from 1 to 0 . 0.10 samples 10% from Orginial data. Set this to 1 for spliting full dataset
when sample_size is set to 1 then use updated_df for train_test_split
"""
SAMPLE_SIZE = 0.10

def create_df_csv(dataframe,filename):
    dataframe.to_csv(filename, index=False)

updated_df = pd.read_csv("Updated_Data_Entry_2017.csv")
print("Length of Original  Dataset : " , len(updated_df))

sample_df = updated_df.sample(frac=SAMPLE_SIZE, replace=False, random_state=1)
print("Length of Sample  Dataset : " , len(sample_df))


train, test = train_test_split(sample_df, test_size=0.2, random_state=42, shuffle=True)
#train, test = train_test_split(updated_df, test_size=0.2, random_state=42, shuffle=True)
print("Length of Sample train Dataset : " , len(train))
print("Length of Sample val Dataset : " , len(test))

create_df_csv(train,"train_" + str(SAMPLE_SIZE) + ".csv")
create_df_csv(test,"val_" +  str(SAMPLE_SIZE) + ".csv")


