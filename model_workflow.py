import argparse
import copy
from typing import Tuple

import numpy as np
import pandas as pd
# from data_downloader import Downloader
from model_evaluation import ModelEvaluation
from model_packager import ModelPackager
from model_training import (ModelTraining, inverse_transformations,
                            transformations)
from preprocessing import PreProcessing
# from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler
# from xgboost import XGBRegressor
import json

# from pyspark.sql import SparkSession

# if __name__ == "__main__":
#     if spark_main_opts:
#         # Set main options, e.g. "--master local[4]"
#         os.environ['PYSPARK_SUBMIT_ARGS'] = spark_main_opts + " pyspark-shell"

    # Set spark config
# spark = (SparkSession.builder.getOrCreate())
spark = "prakash"
with open("./qep_capex/config.json", "r") as json_file:
    config_file = json.load(json_file)

training_dict = {
  # "RandomForestRegressor": RandomForestRegressor,
  # "XGBRegressor": XGBRegressor,
  # "LinearRegression": LinearRegression,
  "HuberRegressor": HuberRegressor
}

output_param_grid = {
   "XGBRegressor": {
   "n_estimators": None,
   "learning_rate": None,
   "subsample": None,
   },
  "HuberRegressor": {
   "epsilon": None,
   "alpha": None,
   "tol": None,
   "max_iter": None
   },
}

def train_model(
  basin_of_interest: str,
  reservoirs: Tuple[str],
  config_param: dict,
  z

):

  # download the completions data for a given basin and reservoirs
  if basin_of_interest:
    basin_of_interest = basin_of_interest.upper()

  new_reservoir = []
  for reservoir in reservoirs:
    if reservoir:
      new_reservoir.append(reservoir.upper())


  # df = Downloader(

  #   basin_of_interest=basin_of_interest,
  #   reservoirs_of_interest=tuple(reservoirs),
  #   dataset_completions="produced.vw_well_completions_merged",
  #   dataset_enverus="produced.premerge_enverus_completions",
  #   impute_from_enverus=True
  # ).run(spark)
  # df = pd.read_parquet('./df_filtered.parquet')
  df = pd.read_parquet('./df.parquet')

  # load the configs
  # df.to_parquet('./df.parquet')
  # print(df.head())
  all_run_config = config_param
  run_config = all_run_config.get(basin_of_interest)
  original_run_configs = copy.deepcopy(run_config)

  if not set(reservoirs).issubset(set(run_config["reservoirs_of_interest"])):
    raise AttributeError(f"Reservoirs provoded {reservoirs} are not in training reservoirs {run_config['reservoirs_of_interest']}")

  if not run_config:
    raise AttributeError(f"Config for {basin_of_interest} not available")
  df_filtered = PreProcessing(data=df).run(filtering_dict=run_config.get("filtering_config"))
  # df_filtered.to_parquet('C:\Users\prakash.pandey\Desktop\QEP\qep-ds-capex_model\df_filtered.parquet')
  print(df_filtered.head())
  # initiate the ModelTraining object. The catergorical columns are encoded here if specified in the config.
  transformation_config = run_config.get("transformations")
  # print('before transformation')
  if transformation_config:
    df_filtered = transformations(df_filtered, transformation_config)
  # print('transformation complete')
  # # df.columns = df.columns.str.replace('/', '_')
  # transformation_config = run_config.get("transformations")
  # if transformation_config:
  #   print('transformation starting -------------------')
  #   df = transformations(df, transformation_config)
  #   # print(dff)
  #   print('transformation cmoplete---------------------')
  #   print(df.head())
  # # filter the data based on filtering config
  # # print(list(df.columns))
  # df_filtered = df
  # df_filtered = PreProcessing(data=df).run(filtering_dict=run_config.get("filtering_config"))
  # df_filtered.to_parquet('C:\Users\prakash.pandey\Desktop\QEP\qep-ds-capex_model\df_filtered.parquet')
  # print(df_filtered.head())
  # initiate the ModelTraining object. The catergorical columns are encoded here if specified in the config.

  training_model = ModelTraining(
  data=df_filtered,
  config=run_config
  )
  df_retraining = df_filtered.copy(deep=True)


  # if transformations are specified in the config, transorm variables based on the config.
  # print('transformation is starting ---------------')
  # transformation_config = run_config.get("transformations")
  # if transformation_config:
    # training_model.df = transformations(training_model.df, transformation_config)
    # df_f
  # print('transformation completed --------------')
  print(df_filtered.head())
  # return df_filtered

  training_config = run_config.get("training_config")
  updated_config = {}

  for key in training_config.keys():
    if key not in list(training_dict.keys()):
      raise AttributeError(f"{key} model is not initiated in training_dict.")
    # if the model name is specified in the training dict, replace the model name with the model object.
    updated_config[training_dict[key]] = training_config[key]

  # split the data into test, train and validation data based on the ratios specified.
  _ = training_model.split_data(
    return_test=run_config["split_data"].get("test_data"),
    train_valid_ratio=run_config["split_data"].get("validation_data_ratio"),
    test_ratio=run_config["split_data"].get("test_data_ratio"),
  )

  # scale input(x) and target data(y) based on the ratios specified in config.
  if run_config["scale_data"].get("scale_x") or run_config["scale_data"].get("scale_y"):
    _ = training_model.scale_data(
      scale_x=run_config["scale_data"].get("scale_x"),
      scale_y=run_config["scale_data"].get("scale_y"),
      scaler_x= RobustScaler
    )

  # train the models
  _ = training_model.fit_predict(training_configs=updated_config, store_cv_results=True)


  # if hyperparameter configs are not specified, use the fitted model and prepare outputs.
  hyperparam_tuning_config = run_config.get("hyperparam_tuning_config")
  if hyperparam_tuning_config is None:
    model = training_model.models_evaluated[key]["model"]

    # initiate the Model Evalation object.
    evaluate_base_models = ModelEvaluation(
      model_evaluation_dict=training_model.models_evaluated,
      no_models_selected_for_tuning=1,
      training_configs_used = updated_config
    )
    evaluate_base_models.optimum_model_details = training_model
    evaluate_base_models.compile_optimum_model_dataframes()

    model_performance, packaged_model, meta_data, model_params = prepare_outputs(
       evaluate_base_models,
       evaluate_base_models.base_models_performance_df,
       model,
       training_model,
       basin_of_interest,
       original_run_configs,
       df_retraining
    )

  else:

    tuned_models = ModelEvaluation(
      model_evaluation_dict=training_model.models_evaluated,
      no_models_selected_for_tuning=1,
      training_configs_used = updated_config
    )

    for key in hyperparam_tuning_config.keys():
        hyperparam_tuning_config[key]["model"] = training_model.models_evaluated[key]["model"]

    # if hyperparameter tuning configs are specified, use RandomizedSearchCV to find optimum hyperparameters.
    tuned_models.tune_hyperparameters(
      hyperparamter_grid=hyperparam_tuning_config,
      trained_models=training_model,
      tuning_method=RandomizedSearchCV
    )
    model = tuned_models.optimum_model_details.models_evaluated[list(tuned_models.optimum_model_details.models_evaluated.keys())[0]]["model"]

    model_performance, packaged_model, meta_data, model_params = prepare_outputs(
       tuned_models,
       tuned_models.optimum_models_performance_df,
       model,
       training_model,
       basin_of_interest,
       original_run_configs,
       df_retraining
    )
  print(model_performance)
  return model_performance, packaged_model ,training_model.operator_list, training_model.res_operator_list, meta_data, model_params, training_model


def prepare_outputs(
    trained_models: ModelEvaluation,
    performance_df: pd.DataFrame,
    model: callable,
    training_model: ModelTraining,
    basin_of_interest: str,
    run_config: dict,
    df: pd.DataFrame
  ):

  # extract feature importance for the final model.
  feature_importance = trained_models.plot_feature_importance(model_name=list(trained_models.optimum_model_details.models_evaluated.keys())[0], return_list_importance=True)

  # create dictionary of expected variance.
  errors = get_error_dict(scaler_x=training_model.scaler_x, df=trained_models.training_results_df, transformation_config=run_config.get("transformations"))

  # create model performance dictionary.
  model_performance = {
      "overall_performance": performance_df[performance_df["models"] == type(model).__name__].to_dict(orient="list"),
      #"operator_list": training_model.operator_list,
      "feature_importance": feature_importance,
      "test_performance_df": trained_models.test_results_df,
      "validation_performance_df": trained_models.validation_results_df,
      "error_results": {"error_dict": errors[0],
                        "error_bins_lateral_length": errors[1],
                        "error_bins_proppant": errors[2],
                        "error_bins_fluid": errors[3]
      }
  }

  # initiate model packager object.
  packaged_model = ModelPackager(
    model=model,
    model_name=type(model).__name__,
    training_model_details=training_model,
    basin=basin_of_interest,
    config=run_config
  )

  # Extract training meta data.
  meta_data = packaged_model.model_metadata(
    filtered_config=run_config.get("filtering_config"),
    hyperparameter_tuning_config=run_config.get("hyperparam_tuning_config")
  )

  model_name = trained_models.models_used_for_tuning[0]

  if model_name not in output_param_grid.keys():
    model_params = {"name": model_name}
  else:
    model_config = training_model.models_evaluated[model_name]["model"].__dict__
    model_params = {"name": model_name}
    for key in output_param_grid[model_name].keys():
        model_params[key] = model_config.get(key)

  # Tranform input data and use _package functionality to fit the final pipeline.
  transformation_config = run_config.get("transformations")
  if transformation_config:
    df = transformations(df, transformation_config)
  input_training_data, input_target_data = split_data(df, run_config)
  pipline = packaged_model._package(run_config, input_data=input_training_data, target_data=input_target_data)

  return model_performance, pipline, meta_data, model_params


def get_error_dict(scaler_x: callable, df: pd.DataFrame, transformation_config: dict):
  # get list of columns from the training dataframe
  columns = [col for col in df.columns if not(col.startswith('y_') or col.startswith('residual_'))]
  ycols = [col for col in df.columns if (col.startswith('y_') or col.startswith('residual_'))]

  # inverse transform the data from the dataframe
  print('----------------inverse1st')
  df_t = pd.DataFrame(scaler_x.inverse_transform(df[columns].values), columns=columns)
  print('------------------inverse2')
  df_t = df_t.join(df[ycols])

  # calculate MAPE
  df_t["MAPE"] = 100 * abs(df_t[[col for col in ycols if col.startswith('residual_')][0]]) / df_t["y_train"]

  if transformation_config is not None:
    print('inside if --------------- inverse')
    print(type(df_t), df_t.shape)
    print(df_t.columns)
    df_t = inverse_transformations(df_t, transformation_config)
  print('inverse3----------------------------')
  df_copy = df_t.copy(deep=True)

  # split proppant intensity in the steps of 200 #/ft
  prop_steps = ((df_copy["Proppant_LBSPerFT"].max()  - df_copy["Proppant_LBSPerFT"].min()) // 200) * 10

  # split fluid intensity in the steps of 10 bbl/ft
  fluid_steps = ((df_copy["Fluid_BBLPerFT"].max()  - df_copy["Fluid_BBLPerFT"].min()) // 10)

  # create bins of lateral length in the steps of 1000ft
  df_t["bins_l"], bins_ll, bin_name = create_bins(df_t, step=1000, property="LateralLength_FT")
  d = {}
  for bin in bin_name:
    df_prop = df_t[df_t["bins_l"] == bin]
    if df_prop.empty:
        continue

    # for each bin in lateral length bins, create proppant intensity bins.
    df_prop["bins_prop"], _, bin_name_prop = create_bins(df_prop, step=prop_steps, property="Proppant_LBSPerFT")
    d[bin] = {}

    for bin_prop in bin_name_prop:
      df_fluid = df_prop[df_prop["bins_prop"] == bin_prop]
      if df_fluid.empty:
          continue

      # for each bin in proppant intensity bins, create fluid intensity bins.
      df_fluid["bins_fluid"], _, _ = create_bins(df_fluid, step=fluid_steps, property="Fluid_BBLPerFT")
      d[bin][bin_prop] = df_fluid.groupby("bins_fluid")["MAPE"].median().round(2).to_dict()

  prop_bins = list(np.arange(prop_steps * (df_copy["Proppant_LBSPerFT"].min() // prop_steps), prop_steps * (2 + (df_copy["Proppant_LBSPerFT"].max() // prop_steps)), prop_steps))
  fluid_bins = list(np.arange(fluid_steps * (df_copy["Fluid_BBLPerFT"].min() // fluid_steps), fluid_steps * (2 + (df_copy["Fluid_BBLPerFT"].max() // fluid_steps)), fluid_steps))

  return d, bins_ll, prop_bins, fluid_bins


def split_data(df: pd.DataFrame, config: dict):

  input_features = config.get("input_features")

  if config.get("encode"):
    for cat in config.get("categorical_feature"):
      input_features.append(cat)

  X = df[input_features]
  y = df[config.get("target_features")]

  if config.get("split_data").get("test_data"):
    test_ratio = config.get("split_data").get("test_data_ratio")
    train_valid_ratio = config.get("split_data").get("validation_data_ratio")

    if test_ratio:
        test_ratio = 0.1

    X, _, y, _ = train_test_split(
        X, y, random_state=42, test_size=test_ratio
    )

  X_train, _, y_train, _ = train_test_split(
      X, y, random_state=42, test_size=train_valid_ratio
  )

  return X_train, y_train

def create_bins(df: pd.DataFrame, step: int, property: str):

  bins = list(np.arange(step * (df[property].min() // step), step * (2 + (df[property].max() // step)), step))
  bin_name = [f"{int(bins[i-1])} - {int(bins[i])}" for i in range(1, len(bins))]
  return pd.cut(df[property], bins, labels=bin_name), bins, bin_name


def model_predict(
  input_data: pd.DataFrame,
  pipeline: callable,
  error_results: dict = None,
  drilling_cost_multiplier: float = 0.325,
  completions_cost_multiplier: float = 0.625,
  facilities_cost_multiplier: float = 0.05,

):
  input_data_copy = input_data.copy(deep=True)
  total_cost = pipeline.predict(input_data)

  results = {
     "TotalCost_USDMM": np.round(total_cost, 2),
     "Drilling_USDMM": total_cost * drilling_cost_multiplier,
     "Facilities_USDMM": total_cost * facilities_cost_multiplier,
     "Completions_USDMM": total_cost * completions_cost_multiplier,
  }

  if error_results:
    bins_ll = error_results.get("error_bins_lateral_length")
    bin_name_ll = [f"{int(bins_ll[i-1])} - {int(bins_ll[i])}" for i in range(1, len(bins_ll))]

    bins_prop = error_results.get("error_bins_proppant")
    bin_name_prop = [f"{int(bins_prop[i-1])} - {int(bins_prop[i])}" for i in range(1, len(bins_prop))]

    bins_fluid = error_results.get("error_bins_fluid")
    bin_name_fluid= [f"{int(bins_fluid[i-1])} - {int(bins_fluid[i])}" for i in range(1, len(bins_fluid))]

    input_data_copy["bins_ll"] = pd.cut(input_data_copy["LateralLength_FT"], bins_ll, labels=bin_name_ll)
    input_data_copy["bins_prop"] = pd.cut(input_data_copy["Proppant_LBSPerFT"], bins_prop, labels=bin_name_prop)
    input_data_copy["bins_fluid"] = pd.cut(input_data_copy["Fluid_BBLPerFT"], bins_fluid, labels=bin_name_fluid)

    error_list = []
    for _, row in input_data_copy.iterrows():
      prop = error_results.get("error_dict").get(row["bins_ll"], None)
      if prop is None:
        error_list.append(np.nan)
        continue

      fluid = prop.get(row["bins_prop"], None)
      if fluid is None:
        error_list.append(np.nan)
        continue

      error = fluid.get(row["bins_fluid"], np.nan)

      error_list.append(error)


  results["expected_cost_variance"] = error_list


  return results


def basin_name(basin):
    return basin.upper()


def reservoirs(reservoirs):
    return tuple([str(i).upper() for i in reservoirs.split(',')])

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--basin', type=basin_name, default = 'ANADARKO EAST',
    #                     help='Name the basin to use: Must be in this list [Appalachia, DJ Basin, Gulf Coast West, Gulf Coast East, Anadarko, Williston, Midland, Delaware]')

    # parser.add_argument('--reservoir', type=reservoirs, default = ['DEESE','LIME', 'HEALDTON','CLEVELAND','PENNINGTON','INOLA','TUSSY','GRANITE WASH'],
    #                     help='List the reservoirs that needs to be used for training seperate by comma ex. Bossier, Haynesville')

    # parser.add_argument('--target_variable', type=str, default = "TotalWellCost_USDMM",
    #                     help='Target variable for the machine learning model. Default is TotalWellCost_USDMM')

    # parser.add_argument('--encode', type=reservoirs, default = False,
    #                     help='One hot encode Reservoirs and use them in training? Default is False')

    # args = parser.parse_args()

    return train_model("ANADARKO EAST", ["WOODFORD","MISSISSIPPIAN"], config_file, spark)

if __name__ == '__main__':
   main()
