import warnings

import flwr as fl
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import flcore.datasets as datasets
from flcore.results import save_pipeline_inference
from flcore.serialization_funs import serialize_RF, deserialize_RF
import flcore.models.random_forest.utils as utils
from flcore.metrics import calculate_metrics, fit_metrics_server_report, getFairnessResults, visualization_distributed_metrics_server_report
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
)
import time
import flcore.featureselection as fs

# Define Flower client
class MnistClient(fl.client.Client):
    def __init__(self, data,client_id,config):
        self.client_id = client_id
        n_folds_out= config['num_rounds']
        self.seed= config['seed']
        # Load data
        (self.X_train, self.y_train), (self.X_test, self.y_test),self.pipeline = data


        #If fairness is defined enable fairness save the features of X_test 
        #as we need to compute the metrics and drop them from X_train and X_test
        self.enabled_fairness = config["enabled_fairness"]
        if('fairness_attribs' in config):
            self.enabled_fairness = True
            fairness_attribs_names = config["fairness_attribs"]
            self.value_privileged_attrib = config["value_privileged_attrib"]
            self.fairness_attribs =  [col for col in fairness_attribs_names if col in self.X_test.columns]
            # Save the columns if they exist in X_test to make the fairness metrics
            self.fairness_columns_test_values = self.X_test[self.fairness_attribs].copy()
            #In order of the weighting aggregation considering fairness, we also need to keep the training columns for fairness 
            self.fairness_columns_train_values = self.X_train[self.fairness_attribs].copy()
            #drop fairness attributes if you do not want to use them in the training
            if(config["drop_fairness_attribs"]):
                self.X_train = self.X_train.drop(columns= self.fairness_attribs)
                self.X_test = self.X_test.drop(columns= self.fairness_attribs)
            print("Fairness enabled:")
            print("Existing fairness attributes in the tabular data: ", self.fairness_attribs)
            print("The priviledged value for all the fairness attributes is: ", self.value_privileged_attrib)
            print("The dropping of the attributes in the training is enabled or not (1 drops and 0 does not): ", config["drop_fairness_attribs"])



        #If feature selection enable the flag otherwise disable and select all features
        #Important the order to be after fairness just in case there is drop of protected attributes
        if('internal_fs' in config and config['internal_fs'] >0):
            self.num_features = config['internal_fs']
            self.enabled_fs = True
        else:
            self.enabled_fs = False
            self.selected_features_names = self.X_train.columns

        self.splits_nested  = datasets.split_partitions(n_folds_out,0.2, self.seed, self.X_train, self.y_train)
        print("The outcome is: ", self.y_test.name)
        self.bal_RF = config['random_forest']['balanced_rf']
        self.model = utils.get_model(self.bal_RF,self.seed) 
        # Setting initial parameters, akin to model.compile for keras models
        #utils.set_initial_params_client(self.model,self.X_train, self.y_train)

    #To initialize the server (one random client is used to start)
    def get_parameters(self, ins: GetParametersIns):  # , config type: ignore
        #use whatever paramater to send to the server to start communication
        params = [None] #utils.get_model_parameters(self.model)
        #Serialize to send it to server
        #It is forced to send an bytesIO
        parameters_to_ndarrays_final = serialize_RF(params)

        # Build and return response 
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters_to_ndarrays_final,
        )

    def fit(self, ins: FitIns):  # , parameters, config type: ignore
        #parameters = ins.parameters
        ##Deserialize to get the real parameters
        #parameters = deserialize_RF(parameters)
        #utils.set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs

        ###############################################################################################
        #If it is enabled feature selection select the best features for the current client in round 0
        #otherwise return all column names (an empty array does not work. Check why)
        #We do it like this otherwise we need to add param in server and be consequent in the following client new calls
        if(ins.config['server_round']==0):
            metrics = {}
            if(self.enabled_fs == True):
                print("Feature selection is enable:")
                print("Feature selection is sending the top best of the client only")
                fs_topnames = fs.selectKBestfeatures(self.X_train, self.y_train,self.num_features,self.seed)
                parameters_updated = serialize_RF(fs_topnames)
                metrics['enabled_fs'] = 1

                # Build and return response with the top N features
                status = Status(code=Code.OK, message="Success")
                return FitRes(
                    status=status,
                    parameters=parameters_updated,
                    num_examples=0,
                    metrics=metrics
                )
            else:
                print("Feature selection is NOT enabled")
                parameters_updated = serialize_RF(self.X_train.columns)
                metrics['enabled_fs'] = 0

                # Build and return response that no feature selection is performed
                status = Status(code=Code.OK, message="Success")
                return FitRes(
                    status=status,
                    parameters=parameters_updated,
                    num_examples=0,
                    metrics=metrics
                )

                

        #################################################################################################

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_idx, val_idx = next(self.splits_nested)
            X_train_2 = self.X_train.iloc[train_idx, :]
            X_val = self.X_train.iloc[val_idx,:]
            y_train_2 = self.y_train.iloc[train_idx]
            y_val = self.y_train.iloc[val_idx]
            #To implement the center dropout, we need the execution time
            start_time = time.time()
            #restart the model otherwise the estimators grows every aggregation
            self.model = utils.get_model(self.bal_RF,self.seed) 
            self.model.fit(X_train_2[self.selected_features_names], y_train_2)
            #We save the variables for inference
            self.model = save_pipeline_inference(self.model,self.pipeline,X_train_2,self.selected_features_names)
            #self.model.pipeline_processing_ = self.pipeline
            #self.model.pipeline_processing_.fit(X_train_2)
            #self.model.selected_features_names_ = self.selected_features_names
            #accuracy = model.score( X_test, y_test )
            # accuracy,specificity,sensitivity,balanced_accuracy, precision, F1_score = \
            # measurements_metrics(self.model,X_val, y_val)
            y_pred = self.model.predict(X_val[self.selected_features_names])
            metrics = calculate_metrics(y_val, y_pred)
    
            if ins.config['server_round']==1:
                print(f"MANUAL_CHECK: Client={self.client_id} with VALIDATION DATA | "
                    f"Num_Features={len(self.selected_features_names)} | "
                    f"Bal_Acc={metrics['balanced_accuracy']:.4f}")

            elapsed_time = (time.time() - start_time)
            fit_metrics_server_report(metrics,self.model,self.X_test[self.selected_features_names],self.y_test,elapsed_time,self.client_id)
            ####for the new smoothing weight metric, we need to include the fairness information
            if self.enabled_fairness:
                self.model = utils.add_fairness_metrics_to_model(metrics, self.model, \
                                                                     self.fairness_attribs,self.value_privileged_attrib, self.fairness_columns_train_values, \
                                                                     y_val,val_idx,y_pred)
                self.model = utils.add_fairness_metrics_to_Trees_from_RF(metrics, self.model, \
                                                                     self.fairness_attribs,self.value_privileged_attrib, self.fairness_columns_train_values, \
                                                                     X_val[self.selected_features_names], y_val,val_idx)
                
                self.model = utils.add_performance_metrics_to_Trees_from_RF(metrics, self.model,X_val[self.selected_features_names], y_val)
            ###end for the new weight metric

            print(f"num_client {self.client_id} has an elapsed time {elapsed_time}")
            
        print(f"Training finished for round {ins.config['server_round']}")

        # Serialize to send it to the server
        params = utils.get_model_parameters(self.model)
        parameters_updated = serialize_RF(params)

        #print(f"Number of estimators: {len(self.model.estimators_)}")

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.X_train),
            metrics=metrics,
        )
        

    def evaluate(self, ins: EvaluateIns):  # , parameters, config type: ignore
        parameters = ins.parameters
        #Deserialize to get the real parameters
        parameters = deserialize_RF(parameters)

        ####################################################################
        #If it is enabled feature selection overwrite the features selected that by default are all the columns
        #If the feature selection is not enabled all the columns are selected by default in the init of the client
        if(ins.config["server_round"]==0 ):
            if(self.enabled_fs == True):
                print("Feature selection is enabled:")
                print("Feature selection is aggregted in the evaluate of the client")
                #here we already have the aggregation of the most important features of all clients and we will
                #select those ones in the dataset
                self.feature_importance = parameters
                            
                # Extract the feature names 
                self.selected_features_names  = [param[0] for param in parameters]

            status = Status(code=Code.OK, message="Success")
            return EvaluateRes(
                status=status,
                loss=0,
                num_examples=0,
                metrics= {}
            )
        
        ####################################################################


        #Always the last function is evulate after fit and we want the last model of server to have it in the client
        #so overwrite the current model
        self.model  = utils.set_model_params(self.model, parameters)
        #print(f"Number of estimators: {len(self.model.estimators_)}")
        #self.model.pipeline_processing_ = self.pipeline
        #self.model.pipeline_processing_.fit(self.X_train)
        #self.model.selected_features_names_ = self.selected_features_names
        #We save the variables for inference
        self.model = save_pipeline_inference(self.model,self.pipeline,self.X_train,self.selected_features_names)
        y_pred_prob = self.model.predict_proba(self.X_test[self.selected_features_names])
        loss = log_loss(self.y_test, y_pred_prob)
        # accuracy,specificity,sensitivity,balanced_accuracy, precision, F1_score = \
        # measurements_metrics(self.model,self.X_test, self.y_test)
        y_pred = self.model.predict(self.X_test[self.selected_features_names])
        metrics = calculate_metrics(self.y_test, y_pred)
        # print(f"Accuracy client in evaluate:  {accuracy}")
        # print(f"Sensitivity client in evaluate:  {sensitivity}")
        # print(f"Specificity client in evaluate:  {specificity}")
        # print(f"Balanced_accuracy in evaluate:  {balanced_accuracy}")
        # print(f"precision in evaluate:  {precision}")
        # print(f"F1_score in evaluate:  {F1_score}")

        visualization_distributed_metrics_server_report(metrics,y_pred_prob,y_pred,self.y_test,self.model,self.X_test[self.selected_features_names],self.client_id )
        if self.enabled_fairness:
            getFairnessResults(metrics,self.y_test.name, y_pred,self.fairness_attribs,self.value_privileged_attrib,\
                    pd.concat([self.y_test, self.fairness_columns_test_values], axis=1))

        # Serialize to send it to the server
        #params = get_model_parameters(model)
        #parameters_updated = serialize_RF(params)
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.X_test),
            metrics=metrics
        )


def get_client(config,data,client_id) -> fl.client.Client:
    return MnistClient(data,client_id,config)
    # # Start Flower client
    # fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
