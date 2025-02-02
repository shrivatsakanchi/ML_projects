import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split



from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

from src.logger import logging
from src.utils import evaluate_models,save_object
logging.info("All are imported")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")
logging.info("Pickel created")
    
@dataclass
class ModelTrainer:
    #model_trainer_config: 'ModelTrainerConfig'
    
    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()
        logging.info("Model tarainer to be started")
   
    def initiate_model_trainer(self,train_array,test_array):
        print("Calling initiate_model_trainer method...")
        
        try:
            print("Started..Loading....")
            logging.info("Splitting traninig, test data")
            X_train,y_train,X_test,y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
           # print(f"Shape of y_train: {y_train.shape}")
            #print(f"Shape of y_test: {y_test.shape}")
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),  
                "AdaBoost Regressor": AdaBoostRegressor(),
            }


            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test = X_test,y_test=y_test,models = models,param=params)

            # to get the best models 
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models [best_model_name]
            if best_model_score <  0.6:
                raise CustomException("No best model")
            logging.info("best model based on training and test data ")

            save_object( file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model)
            
            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            print(f"Error occurred in initiate_model_trainer: {e}")
            raise CustomException(e,sys)
        


#model_trainer_config = ModelTrainerConfig()
#trainer = ModelTrainer(model_trainer_config=model_trainer_config)
#train_array,test_arrray = train_test_split(df,test_size=0.2,random_state=42)
#trainer.initiate_model_trainer(train_array, test_array)
            


