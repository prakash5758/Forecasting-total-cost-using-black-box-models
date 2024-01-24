import copy
from typing import Callable

import numpy as np
import pandas as pd
from model_training import ModelTraining, transformations, operator_threshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class ModelPackager:

  def __init__(
      self, model:
      Callable,
      model_name: str,
      training_model_details: ModelTraining,
      config: dict,
      basin: str
  ):

    self.model = model
    self.training_model_details = training_model_details

    self.scaler_x = training_model_details.scaler_x if training_model_details.is_scaled_x else None
    self.scaler_y = training_model_details.scaler_y if training_model_details.is_scaled_y else None
    self.encoder = training_model_details.encoder if training_model_details.encoder else None

    self.basin = basin
    self.reservoir = config.get("reservoirs_of_interest")
    self.model_name = model_name

    self.metadata = {}

  def _package(self, config: dict, input_data: pd.DataFrame, target_data: np.ndarray):

    numeric_transformer = None
    categorical_transformer_reservoir = None
    categorical_transformer_operator = None

    categorical_features = config.get('categorical_feature')
    numeric_features = config.get('input_features')
    if categorical_features:
      for cat in categorical_features:
        if cat in numeric_features:
          numeric_features.remove(cat)

    if self.scaler_x is not None:
      numeric_transformer = Pipeline(
          steps=[("scaler",self.scaler_x)]
      )

    if self.encoder:
      for key, encoder in self.encoder.items():
        if key == "ReservoirGoldConsolidated":
          categorical_transformer_reservoir = Pipeline(
              steps=[
                  ("encoder", self.encoder[key])
              ]
          )
        if key == "OperatorGold":
          categorical_transformer_operator = Pipeline(
              steps=[
                  ("encoder", self.encoder[key])
              ]
          )


    if numeric_transformer and self.encoder:
      if all([categorical_transformer_reservoir, categorical_transformer_operator]):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat_reservoir", categorical_transformer_reservoir, ["ReservoirGoldConsolidated"]),
                ("cat_operator", categorical_transformer_operator, ["OperatorGold"]),
            ]
        )

      elif categorical_transformer_reservoir:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat_reservoir", categorical_transformer_reservoir, ["ReservoirGoldConsolidated"]),
            ]
        )

      elif categorical_transformer_operator:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat_operator", categorical_transformer_operator, ["OperatorGold"]),
            ]
        )


      pipe = Pipeline(
          steps=[("preprocessor", preprocessor), ("model", self.model)]
      )

    elif numeric_transformer:
      pipe = Pipeline(
          steps=[("scaler", numeric_transformer), ("model", self.model)]
      )

    elif categorical_transformer_operator and categorical_transformer_operator:
      pipe = Pipeline(
          steps=[
                ("cat_reservoir", categorical_transformer_reservoir, ["ReservoirGoldConsolidated"]),
                ("cat_operator", categorical_transformer_operator, ["OperatorGold"]), ("model", self.model)]
      )

    elif categorical_transformer_reservoir:
      pipe = Pipeline(
          steps=[
                ("cat_reservoir", categorical_transformer_reservoir, ["ReservoirGoldConsolidated"]), ("model", self.model)]
      )

    elif categorical_transformer_operator:
      pipe = Pipeline(
          steps=[("cat_operator", categorical_transformer_operator, ["OperatorGold"]), ("model", self.model)]
      )

    else:
      pipe = Pipeline(
          steps=[("model", self.model)]
      )

    pipe.fit(input_data, target_data.values)

    return pipe

  def model_metadata(self, filtered_config: dict, hyperparameter_tuning_config: dict):

    training_config = self.training_model_details.training_spec

    param_grid = None
    model_hyperparameter_data= None

    if hyperparameter_tuning_config:
      model_hyperparameter_data = hyperparameter_tuning_config.get(
        self.model_name, None
      )

    if model_hyperparameter_data:
      param_grid= model_hyperparameter_data.get("param_grid", None)

    self.metadata = {
      "filtereing_spec":{
      "columns_used_for_filtering": filtered_config.get("cols_to_keep", None),
      "technical_filtering_thresholds":filtered_config.get("technical_filtering", None),
      "statistical_filtering":filtered_config.get("statistical_filtering", None),
      "ratio_filtering":filtered_config.get("ratio_filtering", None),
      },

      "input_features": self.training_model_details.input,
      "output_feature": self.training_model_details.target,

      "split_spec": training_config.get("split_spec", None),
      "scaling_spec": training_config.get("scaling_spec", None),

      f"hyperparameter_tuning_spec_{self.model_name}": param_grid,
    }

    return self.metadata