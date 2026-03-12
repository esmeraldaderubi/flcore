import argparse
import os

def get_parser(isserver):
    parser = argparse.ArgumentParser(description="Generate config dictionary from arguments")

    # General arguments
    # Possible values: youthgems_format, kaggle_hf, mnist, dt4h_format
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_port", type=int, default=8081)
    parser.add_argument("--production_mode", type=bool, default=True)
    parser.add_argument("--num_rounds", type=int, default=50)
 
    #Model arguments
    parser.add_argument("--model", choices=["logistic_regression", "lsvc", "elastic_net", "random_forest", "weighted_random_forest", "xgb"], required=True)
    parser.add_argument("--data_path", default="dataset/")
    parser.add_argument("--outcome", default="Eval")
    parser.add_argument("--feature_subset", nargs="+") 
    
    # Model-specific args (optional, validated later)
    parser.add_argument("--n_features", type=int, default=0) ###DEPRECATED BY internal_fs FIX!!
    parser.add_argument("--balanced_rf", type=bool, default=True)
    parser.add_argument("--aggregator_rf",default="randomviaprobs",choices=["all", "random", "randomviaprobs", "randomviaprobswithTradeoffMetrics", "sortedTradeoffMetrics","totalsortedTradeoffMetrics","ParetoQuota","DiversityPareto","AdaptiveDiversitySizeCenterProbs", "AdaptiveDiversitySizeCenterProbsv2","FairnessUtilityQuota"])
    parser.add_argument("--smoothedWeights_baseline_type",default="equal_voting",choices=["equal_voting", "fedavg"])
    parser.add_argument("--beta_fairness_trade_off", type=float, default=0.75)
    
    parser.add_argument("--levelOfDetail",default="RandomForest",choices=["DecisionTree", "RandomForest"])
    parser.add_argument("--batch_size", type=int,default=32)
    parser.add_argument("--num_iterations", type=int,default=100)
    parser.add_argument("--task_type",default="BINARY")
    parser.add_argument("--tree_num", type=int,default=500)  
    
    # Other config
    #parser.add_argument("--held_out_center_id", type=int, default=-1)
    
    #specific variables for client and server 
    if(isserver == True): #server
        parser.add_argument("--num_clients", type=int, default=1)
        #parser.add_argument("--checkpoint_selection_metric", choices=["accuracy", "balanced_accuracy", "f1", "precision", "recall"], required=True)
        # Dropout and smoothing
        parser.add_argument("--dropout_method", default="None")
        parser.add_argument("--percentage_drop", type=int, default=50)
        parser.add_argument("--smooth_method", default="None",choices=["None","EqualVoting", "fairnessWeighting", "SlowerQuartile", "SupperQuartile"])
        parser.add_argument("--smoothing_strenght", type=float, default=0.5)
    
    else:     
        parser.add_argument("--name_client", default=os.getenv("NODE_NAME"))
        parser.add_argument("--fairness_attribs", nargs="+")
        parser.add_argument("--value_privileged_attrib", type=int, default=1)
        parser.add_argument("--drop_fairness_attribs", type=int, default=1)
        parser.add_argument("--internal_fs", type=int, default=0)
        
    return parser

def validate_model_specific_args(args):
    model_args = {
        "logistic_regression": ["n_features"],
        "lsvc": ["n_features"],
        "elastic_net": ["n_features"],
        "random_forest": ["balanced_rf","aggregator_rf"],
        "weighted_random_forest": ["balanced_rf", "levelOfDetail"],
        "xgb": ["batch_size", "num_iterations", "task_type", "tree_num"]
    }
    allowed_args = model_args.get(args.model, [])
    passed_args = [k for k, v in vars(args).items() if v is not None]
    for arg in passed_args:
        if arg in ["n_features", "balanced_rf","aggregator_rf","smoothedWeights_baseline_type", "beta_fairness_trade_off", "smoothing_strenght", "levelOfDetail", "batch_size", "num_iterations", "task_type", "tree_num"]:
            if arg not in allowed_args:
                raise ValueError(f"Argument --{arg} is not allowed for model '{args.model}'")

def generate_config_dict(args, isserver):
    config_dict = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "local_port": args.local_port,
        "data_path": args.data_path,
        "production_mode": args.production_mode,
        "outcome" : args.outcome,
        "num_rounds" :  args.num_rounds,
        
    }


    # If a subset of features is defined
    if hasattr(args, "feature_subset") and args.feature_subset is not None:
        config_dict["feature_subset"] = args.feature_subset

    if(isserver):
        config_dict["num_clients"] = args.num_clients
        config_dict["dropout_method"] =  args.dropout_method
        config_dict["dropout"] = {"percentage_drop" : args.percentage_drop}
        config_dict["smooth_method"] =  args.smooth_method
        config_dict["smoothWeights"] = {"smoothing_strenght" : args.smoothing_strenght}
    else:
        config_dict["name_client"] = args.name_client    
        config_dict["internal_fs"] = args.internal_fs
        # If fairness is defined
        if hasattr(args, "fairness_attribs"):
            config_dict["fairness_attribs"] = args.fairness_attribs
            config_dict["enabled_fairness"] = True
            #This version all the protected variables have the same privileged value
            config_dict["value_privileged_attrib"] = args.value_privileged_attrib
            #drop the protected variables for the training or not
            config_dict["drop_fairness_attribs"] = args.drop_fairness_attribs

        else:
            config_dict["enabled_fairness"] = False        
            #You cannot enable fairnessWeighting if fairness attributes are not defined!!
            if( args.smooth_method== "fairnessWeighting"):
                raise ValueError(f"Argument --{args.smooth_method} cannot be used  if fairness attributes are not defined")
             #You cannot enable randomviaprobswithTradeoffMetrics OR sortedTradeoffMetrics if fairness attributes are not defined!!
            if(args.aggregator_rf== 'randomviaprobswithTradeoffMetrics' or args.aggregator_rf == 'sortedTradeoffMetrics'):   
                raise ValueError(f"Argument --{args.aggregator_rf} cannot be used  if fairness attributes are not defined")


    # Add model-specific fields
    if args.model in ["logistic_regression", "lsvc", "elastic_net"]:
        config_dict["linear_models"] = {"n_features": args.n_features}
    if args.model == "random_forest" :
        config_dict["random_forest"] = {
            "balanced_rf": args.balanced_rf,
            "aggregator_rf": args.aggregator_rf}

    if args.model == "weighted_random_forest":
        config_dict["weighted_random_forest"] = {
            "balanced_rf": args.balanced_rf,
            "levelOfDetail": args.levelOfDetail
        }
    if args.model == "xgb":
        config_dict["xgb"] = {
            "batch_size": args.batch_size,
            "num_iterations": args.num_iterations,
            "task_type": args.task_type,
            "tree_num": args.tree_num
        }

    return config_dict