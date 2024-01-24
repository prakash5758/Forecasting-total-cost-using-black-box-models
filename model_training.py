import copy
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import (GridSearchCV, cross_validate,
                                     train_test_split)
from sklearn.preprocessing import OneHotEncoder, StandardScaler


tranformation_map = {
    "log": np.log,
    "square": np.square,
    "sqrt": np.sqrt,
    "multiply": np.multiply
}

inverse_map = {
    "log": np.exp,
    "square": np.sqrt,
    "sqrt": np.square,
    "multiply": np.divide
}

class ModelTraining:
    def __init__(
        self,
        data: pd.DataFrame,
        config: dict,
        encoder: Callable = OneHotEncoder,
        input_features: list = None,
        copy_object: bool = False
    ):
        self.config = config
        run_config = copy.deepcopy(config)
        categorical_cols = None
        if run_config.get("encode"):
            categorical_cols=run_config.get("categorical_feature")

        self.df = data
        self.input = run_config.get("input_features")

        if input_features is not None:
            self.input = self.input
        self.target = run_config.get("target_features")[0]

        self.training_data = None
        self.test_data = None
        self.valid_data = None

        self.is_scaled_x: bool = False
        self.scaler_x: Callable = None
        self.scaler_y: Callable = None
        self.is_scaled_y: bool = False
        self.operator_list: list = None
        self.res_operator_list: list = None

        self.training_spec = {}
        self.models_evaluated = {}
        self.hyperparameter_search = {}

        self.encoder = {}
        op_thresh = config.get('operator_threshold', '50')
        if not copy_object:
            if categorical_cols:
                op_list = self.operator_threshold(op_thresh=op_thresh)
                self.operator_list = op_list
                self.df['OperatorGold'] = self.df['ReservoirGoldConsolidated'].astype('str')+'_'+self.df['OperatorGold'].astype('str')
                self.res_operator_list = self.df['OperatorGold'].unique().tolist()
                for cat_col in categorical_cols:
                    if cat_col.lower() == 'operatorgold':
                        col_names = self.categorical_encoding(cat_col, encoder=encoder)
                    else:
                        col_names = self.categorical_encoding(cat_col, encoder=encoder)
                    self.input += col_names

    def split_data(
        self,
        return_test: bool = False,
        train_valid_ratio: float = 0.2,
        test_ratio: float = None,
    ):
        X = self.df[self.input].copy(deep=True)
        y = self.df[self.target].copy(deep=True)

        X_test = y_test = None
        if return_test:
            if not test_ratio:
                test_ratio = 0.1

            X, X_test, y, y_test = train_test_split(
                X, y, random_state=42, test_size=test_ratio
            )

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, random_state=42, test_size=train_valid_ratio
        )

        self.training_data = (X_train, y_train)
        self.test_data = (X_test, y_test)
        self.valid_data = (X_valid, y_valid)

        self.training_spec["split_spec"] = {
            "train_to_valid_ratio": train_valid_ratio,
            "test_split_created": return_test,
            "test_to_train/valid_ratio": test_ratio,
        }

        return {
            "train_data": self.training_data,
            "valid_data": self.test_data,
            "test_data": self.valid_data,
        }


    def operator_threshold(
        self,
        op_thresh: Optional[int] = 50,
        ) -> Optional[list]:

        op_value_count = self.df['OperatorGold'].value_counts()
        op_thresh = op_thresh
        condition = op_value_count < op_thresh
        mask_obs = op_value_count[condition].index
        mask_dict = dict.fromkeys(mask_obs, 'miscellaneous')
        self.df['OperatorGold'] = self.df['OperatorGold'].replace(mask_dict)
        op_list_retained = list(op_value_count[~condition].index)
        return op_list_retained

    def categorical_encoding(
        self,
        categorical_cols: list,
        encoder: Optional[Callable] = OneHotEncoder,
    ) -> Optional[dict]:

        encoder_transform = encoder(sparse=False, drop='first').fit(self.df[[categorical_cols]].values.reshape(-1, 1))

        encoded = encoder_transform.transform(self.df[[categorical_cols]])

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder_transform.get_feature_names_out()
        )

        self.encoder[categorical_cols] = encoder_transform
        self.df = self.df.join(encoded_df)
        self.df.drop(categorical_cols, axis=1, inplace=True)

        return list(encoder_transform.get_feature_names_out())


    def scale_data(
        self,
        scale_x: bool = True,
        scale_y: bool = False,
        scaler_x: Callable = StandardScaler,
        scaler_y: Optional[Callable] = StandardScaler,
    ) -> Optional[dict]:

        if not scale_x and not scale_y:
            self.training_spec["scaling_spec"] = {
                "is_x_scaled": scale_x,
                "is_y_scaled": scale_y,
            }
            return None

        X_train, y_train = self.training_data
        X_valid, y_valid = self.valid_data
        X_test, y_test = self.test_data

        X_test_scaled = y_test_scaled = None
        X_train_scaled = X_train
        X_valid_scaled = X_valid
        X_test_scaled = X_test

        if scale_x:

            self.scaler_x = scaler_x()
            X_train_scaled = self.scaler_x.fit_transform(X_train)
            X_valid_scaled = self.scaler_x.transform(X_valid)

            if X_test is not None:
                X_test_scaled = self.scaler_x.transform(X_test)

            self.is_scaled_x = True

        y_train_scaled = y_train
        y_valid_scaled = y_valid
        y_test_scaled = y_test

        if scale_y:
            self.is_scaled_y = True
            self.scaler_y = scaler_y()
            y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1))
            y_valid_scaled = self.scaler_y.transform(y_valid.values.reshape(-1, 1))

            if y_test is not None:
                y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1))

        self.training_data = (X_train_scaled, y_train_scaled)
        self.test_data = (X_test_scaled, y_test_scaled)
        self.valid_data = (X_valid_scaled, y_valid_scaled)

        self.training_spec["scaling_spec"] = {
            "is_x_scaled": scale_x,
            "x_scaler": scaler_x.__name__,
            "is_y_scaled": scale_y,
            "y_scaler": scaler_y.__name__,
        }

        return {
            "scaled_train_data": self.training_data,
            "scaled_valid_data": self.test_data,
            "scaled_test_data": self.valid_data,
        }

    def fit_predict(
        self, training_configs: Optional[dict] = None, store_cv_results: bool = False
    ):
        if not training_configs:
            raise AttributeError("Training Specifications not provided")

        model_registry = {}
        for model, model_specs in training_configs.items():
            if not model_specs:
                model_specs = {}

            ml_model = None
            ml_model = model(**model_specs)

            ml_model.fit(self.training_data[0], self.training_data[1])
            model_registry[model.__name__] = {"model": ml_model}

            y_pred = ml_model.predict(self.valid_data[0])
            y_test = self.valid_data[1]

            if self.is_scaled_y:
                y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                y_test = self.scaler_y.inverse_transform(y_test.reshape(-1, 1))

            if store_cv_results:
                scores = cross_validate(
                    ml_model,
                    self.valid_data[0],
                    self.valid_data[1],
                    cv=5,
                    scoring=("neg_root_mean_squared_error"),
                )

            model.__name__

            model_performance_Data = {
                "r2": round(r2_score(y_test, y_pred), 5),
                "mean_absolute_percentage_error (%)": round(
                    mean_absolute_percentage_error(y_test, y_pred), 5
                ),
                "root_mean_squared_error (USDMM)": (
                    round(mean_squared_error(y_test, y_pred), 5)
                ) ** 0.5,
                "residual P90/P10": -1 * round(
                    np.quantile(y_test - y_pred, (0.9)) / np.quantile(y_test - y_pred, (0.1)), 5,
                ),
                "residual std (USDMM)": round(np.std(y_test - y_pred), 5),
                "residuals IQR (USDMM)": round(stats.iqr(y_test - y_pred), 5),
                "average cross-validated rmse (USDMM)": -1 * np.average(scores.get("test_score")),
            }

            model_registry[model.__name__]["performance"] = model_performance_Data

        self.models_evaluated = model_registry

        return model_registry

    def hyperparam_tuning(
        self,
        model: Callable,
        param_grid: dict,
        data_to_use: str = "valid",
        cross_valid: float = 4,
        metric: Union[Callable, str] = "neg_mean_absolute_percentage_error",
        method_to_use: callable = GridSearchCV,
        refit: bool = False,
    ):

        if data_to_use == "test":
            X_grid, y_grid = self.test_data
        elif data_to_use == "valid":
            X_grid, y_grid = self.valid_data
        elif data_to_use == "train":
            X_grid, y_grid = self.training_data
        else:
            raise AttributeError(
                f"Invalid data use required {data_to_use}, Must be valid, train, or test"
            )

        hyper_param_search = method_to_use(
            model, param_grid, scoring=metric, cv=cross_valid, n_jobs=-1, refit=refit
        ).fit(X_grid, y_grid)

        self.hyperparameter_search[model] = {
            "best_params": hyper_param_search.best_params_,
            "score": hyper_param_search.best_score_,
            "method": method_to_use.__name__,
        }

        return hyper_param_search.best_params_, hyper_param_search.best_score_

    def create_copy(
        self,
    ):

        new_model_details = ModelTraining(
            data=self.df, config=self.config, input_features=self.input, copy_object=True
        )
        new_model_details.training_data = self.training_data
        new_model_details.valid_data = self.valid_data
        new_model_details.test_data = self.test_data

        new_model_details.input = self.input
        new_model_details.target = self.target

        new_model_details.is_scaled_x = self.is_scaled_x
        new_model_details.scaler_x = self.scaler_x
        new_model_details.scaler_y = self.scaler_y
        new_model_details.is_scaled_y = self.is_scaled_y

        return new_model_details

def operator_threshold(
        op_thresh,
        input_data
):

    op_value_count = input_data['OperatorGold'].value_counts()
    condition = op_value_count < op_thresh
    mask_obs = op_value_count[condition].index
    mask_dict = dict.fromkeys(mask_obs, 'miscellaneous')
    input_data['OperatorGold'] = input_data['OperatorGold'].replace(mask_dict)
    op_list_retained = list(op_value_count[~condition].index)
    return op_list_retained

def transformations(df,transformation_config: dict,order: list = None):

    # return "prakash"

    df_copy = df.copy(deep=True)

    t_config = transformation_config
    if order:
        t_config = {key: transformation_config[key] for key in order if key in transformation_config}

    for feature, transform in t_config.items():
        # if df[feature].dtype != float:
        #     try:
        #         df[feature] = df[feature].astype(float)
        #     except ValueError:
        #         print(f"Column '{feature}' cannot be converted to float.")
        # You can handle the error as per your requirements
        df_copy[feature] = df_copy[feature].astype('float')

        if isinstance(transform, dict):

            transform_key = list(transform.keys())[0]
            df_copy[f"{feature}"] = df_copy.apply(lambda x: tranformation_map[transform_key](x[feature], x[transform[transform_key][0]]), axis=1)
            print(f'------------{feature}-----------')
        else:
            if transform not in tranformation_map.keys():
                raise AttributeError(f"{transform} not avalilable in transformation map")
            else:
                df_copy[f"{feature}"] = df_copy[feature].apply(lambda x: tranformation_map[transform](x))

    return df_copy


# def inverse_transformations(
#             df,
#             transformation_config: dict
#     ):


#     df_copy = df.copy(deep=True)
#     if transformation_config:
#         for feature, transform in transformation_config.items().__reversed__():
#             df_copy[feature] = df_copy[feature].astype('float')

#             if isinstance(transform, dict):
#                 transform_key = list(transform.keys())[0]
#                 df_copy[f"{feature}"] = df_copy.apply(lambda x: inverse_map[transform_key](x[feature], x[transform[transform_key][0]]), axis=1)
#             else:
#                 if transform not in tranformation_map.keys():
#                     raise AttributeError(f"{transform} not avalilable in transformation map")
#                 else:
#                     df_copy[f"{feature}"] = df_copy[feature].apply(lambda x: inverse_map[transform](x))

#     return df_copy

def inverse_transformations(df, transformation_config):
    df_copy = df.copy(deep=True)
    for feature, transform in transformation_config.items():
        if feature in df_copy.columns:
            if transform in inverse_map:
                if isinstance(transform, str):

                    df_copy[feature] = inverse_map[transform](df_copy[feature])
                elif isinstance(transform, dict):
                    transform_key = list(transform.keys())[0]
                    df_copy[feature] = df_copy.apply(lambda x: inverse_map[transform_key](x[feature], x[transform[transform_key][0]]), axis=1)
            else:
                raise ValueError(f"Invalid transformation: {transform}")
        # else:
        #     raise ValueError(f"Column '{feature}' not found in the DataFrame.")
    return df_copy
