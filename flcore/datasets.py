import bz2
import glob
import os
import shutil
import urllib.request
from typing import Tuple
import json

import numpy as np
import openml
#import torch
import pandas as pd

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, f_classif


from flcore.models.xgb.utils import TreeDataset, do_fl_partitioning, get_dataloader
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from imblearn.pipeline import Pipeline as imbPipeline

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]



def define_pipeline():
    #cat_features = data.select_dtypes(include='category').columns
    #length_cats = len(cat_features)
    #print("Data size:",data.shape)
    #print("Number of categorical:", length_cats)
    
    imputer_cat = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    imputer_cont = KNNImputer(n_neighbors=4, weights="uniform")

    numeric_transformer = Pipeline(
        steps=[("imputer", imputer_cont), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", imputer_cat)
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, selector(dtype_exclude="category")),
            ("cat", categorical_transformer, selector(dtype_include="category")),
        ]
    )
    pipeline = imbPipeline([("preprocessor", preprocessor)]).set_output(transform="pandas")

    return pipeline

#It makes the imputer+ scaling in continous
#It makes the imputer in categorical
#Please use data separated by training and testing separately
#and make sure to define the cat and cont initially
def pre_processingyouthgemts(data):
    cat_features = data.select_dtypes(include='category').columns
    length_cats = len(cat_features)
    print("Data size:",data.shape)
    print("Number of categorical:", length_cats)

    pipeline = define_pipeline()
    pipeline.fit(data)

    data = pipeline.transform(data)
    #remove the prefix added by the transformer NUM___ and CAT___
    data.columns = [s[5:] for s in data.columns]

    #add the categorical again
    for col in data.columns:
        if col in cat_features:
            data[col] = data[col].astype('category',copy=False)
        

    return data,pipeline


#The continous values are obtained from the json that has the description of each
#variable
def get_continous_variables(config) -> Dataset:
    data_path = config["data_path"]
    file_name = data_path+"dataset_description.json"

    with open(file_name, "r") as f:
        data = json.load(f)

    # Extract names of variables where type is continuous
    continuous_vars = [item["name"] for item in data if item.get("type") == "continuous"]

    print("Continuous variables:")
    print(continuous_vars)
    return continuous_vars


def select_feature_subsets(df, config) -> Dataset:
    # Check if feature_subset is defined and not None
    if "feature_subset" in config:
        # Only use columns that exist in df to avoid KeyError
        safe_cols = [col for col in config["feature_subset"] if col in df.columns]
        df = df[safe_cols]
    return df

#Select the first file in the folder where the dataset is located
#for debug purposes we allow to choose the position of the file (center_id)
#but in production, the selected file will always be the first file (center_id=0)   
def load_youthgems(config, center_id=0) -> Dataset:
    data_path = config["data_path"]
    continuous_variable_names = get_continous_variables(config) 

    #read the tabular data
    #if center_id == 1:
    #    file_name = data_path+'UKpopulationMentalHealthIssues.csv'
    #else:
    #    file_name = data_path+'WalespopulationMentalHealthIssues.csv'

    try:
        # Check if the folder exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Folder '{data_path}' does not exist.")
        
        # Get all files in the folder
        files = sorted(glob.glob(os.path.join(data_path, "*")))
        
        # If no files found, raise an error
        if not files:
            raise FileNotFoundError(f"No files found in folder '{data_path}'.")

    except FileNotFoundError as e:
        print("Error:", e)
        return None  # Return None to indicate failure

    file_name = files[center_id] # the first file found
    #remove unknown columns
    code_id = "MCSID"
    code_id2 = "ACNUM00"
    code_outcome = config["outcome"]
    print("outcome")
    print(code_outcome)

    data = pd.read_csv(file_name)

    X_data = data.drop([code_id,code_id2, code_outcome], axis=1)
    #If we define a subset of features remove the non-specified ones
    X_data = select_feature_subsets(X_data, config)
    y_data = data[code_outcome]

    
    #f_eid = data[code_id]

    # get column names of data frame in a list
    col_names = list(X_data)
    
    # loop to change each column to category type
    for col in col_names:
        if col not in continuous_variable_names:
            X_data[col] = X_data[col].astype('category',copy=False)


    # Split the data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config["seed"])
    train_index, test_index = next(sss.split(X_data, y_data))
    X_test = X_data.iloc[test_index, :]
    X_train = X_data.iloc[train_index, :]
    y_test, y_train = y_data.iloc[test_index], y_data.iloc[train_index]
    # We save the names
    #f_eid.iloc[test_index]
    #f_eid.iloc[train_index]

    #impute and standarize the data
    X_train,pipeline = pre_processingyouthgemts(X_train)
    X_test, _ = pre_processingyouthgemts(X_test)
    
    print(X_train.shape)
    print(X_test.shape)
    return (X_train, y_train), (X_test, y_test),pipeline



def load_dataset(config, id=0):
    if config["dataset"] == "youthgems_format":
        return load_youthgems(config, id)
    else:
        raise ValueError("Invalid dataset name")

def get_stratifiedPartitions(n_splits,test_size, random_state):
    sss = StratifiedShuffleSplit(n_splits=n_splits,test_size=test_size, random_state=random_state)
    return sss

def split_partitions(n_splits,test_size, random_state,X_data, y_data):
    sss = get_stratifiedPartitions(n_splits,test_size, random_state)
    splits_nested = (sss.split(X_data, y_data))
    return splits_nested
