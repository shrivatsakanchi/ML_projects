import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    prepro_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            num_val = ['reading_score', 'writing_score']
            cat_val = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
        
            num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler",StandardScaler())])  
            cat_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                                           ("one_hot_encoder",OneHotEncoder(sparse_output=True)),
                                           ("scaler",StandardScaler(with_mean=False))]) 
            
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns completed")

            preprocessor = ColumnTransformer([("num_pipeline",num_pipeline,num_val),
                                              ("cat_pipeline",cat_pipeline,cat_val)])
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            logging.info("Read train and test done")
            logging.info("obtaining preprocessing object")
            preprocessing_object = self.get_data_transformer_object()
            target_column  = "math_score"
            num_val = ['reading_score', 'writing_score']
            input_feature_train_df=df_train.drop(columns=[target_column],axis=1)
            target_feature_train_df=df_train[target_column]
            input_feature_test_df=df_test.drop(columns=[target_column],axis=1)
            target_feature_test_df=df_test[target_column]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            save_object(

                file_path=self.data_transformation_config.prepro_obj_file_path,
                obj=preprocessing_object

            )


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.prepro_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)

            
        
                 

