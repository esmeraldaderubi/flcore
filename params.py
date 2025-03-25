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
    parser.add_argument("--n_features", type=int)
    parser.add_argument("--balanced_rf", type=bool)
    parser.add_argument("--levelOfDetail")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_iterations", type=int)
    parser.add_argument("--task_type")
    parser.add_argument("--tree_num", type=int)  
    # Other config
    #parser.add_argument("--held_out_center_id", type=int, default=-1)
    
    #specific variables for client and server 
    if(isserver == True): #server
        parser.add_argument("--num_clients", type=int, default=1)
        #parser.add_argument("--checkpoint_selection_metric", choices=["accuracy", "balanced_accuracy", "f1", "precision", "recall"], required=True)
        # Dropout and smoothing
        parser.add_argument("--dropout_method", default="None")
        parser.add_argument("--percentage_drop", type=int, default=50)
        parser.add_argument("--smooth_method", default="None")
        parser.add_argument("--smoothing_strenght", type=float, default=0.5)
    
    else:     
        parser.add_argument("--name_client", default=os.getenv("NODE_NAME"))
        parser.add_arguments("--fairness_atribs", nargs="+")
        parser.add_arguments("--priviledged_values", type=int, default=1)
        parser.add_arguments("--drop_fairness_attribs", type=int, default=1)

    return parser

def validate_model_specific_args(args):
    model_args = {
        "logistic_regression": ["n_features"],
        "lsvc": ["n_features"],
        "elastic_net": ["n_features"],
        "random_forest": ["balanced_rf"],
        "weighted_random_forest": ["balanced_rf", "levelOfDetail"],
        "xgb": ["batch_size", "num_iterations", "task_type", "tree_num"]
    }
    allowed_args = model_args.get(args.model, [])
    passed_args = [k for k, v in vars(args).items() if v is not None]
    for arg in passed_args:
        if arg in ["n_features", "balanced_rf", "levelOfDetail", "batch_size", "num_iterations", "task_type", "tree_num"]:
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
        "num_rounds" :  args.num_rounds
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
        # If fairness is defined
        if hasattr(args, "fairness_atribs") and args.fairness_atribs is not None:
            config_dict["fairness_atribs"] = args.fairness_atribs
            config_dict["enabledFairness"] = True
        #This version all the protected variables have the same privileged value
        if hasattr(args, "value_privileged_attrib") and args.value_privileged_attrib is not None:
            config_dict["value_privileged_attrib"] = args.value_privileged_attrib
        #drop the protected variables for the training or not
        if hasattr(args, "drop_fairness_attribs") and args.drop_fairness_attribs is not None:
            config_dict["drop_fairness_attribs"] = args.drop_fairness_attribs
            

    # Add model-specific fields
    if args.model in ["logistic_regression", "lsvc", "elastic_net"] and args.n_features is not None:
        config_dict["linear_models"] = {"n_features": args.n_features}
    if args.model == "random_forest" and args.balanced_rf is not None:
        config_dict["random_forest"] = {"balanced_rf": args.balanced_rf}
    if args.model == "weighted_random_forest" and args.balanced_rf is not None and args.levelOfDetail is not None:
        config_dict["weighted_random_forest"] = {
            "balanced_rf": args.balanced_rf,
            "levelOfDetail": args.levelOfDetail
        }
    if args.model == "xgb" and all(v is not None for v in [args.batch_size, args.num_iterations, args.task_type, args.tree_num]):
        config_dict["xgb"] = {
            "batch_size": args.batch_size,
            "num_iterations": args.num_iterations,
            "task_type": args.task_type,
            "tree_num": args.tree_num
        }

    return config_dict