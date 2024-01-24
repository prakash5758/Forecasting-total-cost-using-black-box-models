from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from model_training import ModelTraining
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_dict: dict = None,
        no_models_selected_for_tuning: int = 4,
        training_configs_used: dict = None,
    ):

        self.base_models_performance_df = self.convert_to_performance_df(
            model_evaluation_dict
        )

        self.models_used_for_tuning = (
            self.base_models_performance_df.sort_values(
                by="mean_absolute_percentage_error (%)"
            )
            .models[:no_models_selected_for_tuning]
            .values
        )

        self.original_training_spec = training_configs_used
        self.optimum_models_performance_df = None
        self.optimum_models = None
        self.model_dfs = None

    @staticmethod
    def convert_to_performance_df(model_dict: dict):

        df_results = pd.DataFrame(model_dict).T
        df_performance = pd.DataFrame.from_records(df_results["performance"].values)
        df_performance.index = df_results.index
        df_performance = df_performance.rename_axis("models").reset_index()
        return df_performance.sort_values(by=["mean_absolute_percentage_error (%)"])

    def filter_training_specs(self, model_name: str):

        for key in self.original_training_spec.keys():
            if key.__name__ == model_name:
                return key

    def tune_hyperparameters(
        self,
        hyperparamter_grid: dict,
        trained_models: ModelTraining,
        tuning_method: Callable = GridSearchCV,
        scoring_method: Union[Callable, str] = "neg_mean_absolute_percentage_error",
    ):

        tuning_results = {}
        for model_name, model_tuning_spec in hyperparamter_grid.items():

            tuning_results[model_name] = trained_models.hyperparam_tuning(
                model=model_tuning_spec["model"],
                param_grid=model_tuning_spec["param_grid"],
                method_to_use=tuning_method,
                metric=scoring_method,
                refit=False,
            )

        training_config = {}
        for model_name, performance in tuning_results.items():
            params = performance[0]
            model = self.filter_training_specs(model_name)
            training_config[model] = params

        optimum_model_details = trained_models.create_copy()

        _ = optimum_model_details.fit_predict(
            training_configs=training_config, store_cv_results=True
        )

        self.optimum_models_performance_df = self.convert_to_performance_df(
            optimum_model_details.models_evaluated
        )
        self.optimum_model_details = optimum_model_details
        self.optimum_models = tuning_results

        self.compile_optimum_model_dataframes()


    def compile_optimum_model_dataframes(self):

        self.training_results_df = self.create_dataframe(type="train")
        self.validation_results_df = self.create_dataframe(type="valid")
        self.test_results_df = self.create_dataframe(type="test")


    def create_dataframe(self, type: str = "valid"):

        if type == "train":
            X_data = self.optimum_model_details.training_data[0]
            y_data = self.optimum_model_details.training_data[1]
        elif  type == "valid":
            X_data = self.optimum_model_details.valid_data[0]
            y_data = self.optimum_model_details.valid_data[1]
        elif  type == "test":
            X_data = self.optimum_model_details.test_data[0]
            y_data = self.optimum_model_details.test_data[1]
        else:
            raise(f"Invalid type : {type}. Type of dataframe must be train, valid or test")


        df = pd.DataFrame()

        if self.optimum_model_details.is_scaled_y:
            scaler_y = self.optimum_model_details.scaler_y
            y_data = scaler_y.inverse_transform(y_data).flatten()

            for (model_name, training_data) in self.optimum_model_details.models_evaluated.items():
                df[f"y_pred_{model_name}"] = scaler_y.inverse_transform(
                    training_data["model"].predict(X_data).reshape(-1, 1)
                ).flatten()

        else:

            for (model_name, training_data) in self.optimum_model_details.models_evaluated.items():
                df[f"y_pred_{model_name}"] = training_data["model"].predict(X_data)

        df[f"y_{type}"] = list(y_data)

        for model_name, _ in self.optimum_model_details.models_evaluated.items():
            df[f"residual_{model_name}"] = (df[f"y_{type}"] - df[f"y_pred_{model_name}"])

        return df.join(pd.DataFrame(X_data, columns=self.optimum_model_details.input))



    def inverse_scale_outputs(self, df: pd.DataFrame):

        scale_y = self.optimum_model_details.scaler_y
        cols = [col for col in df.columns if "y_" in col]
        for col in cols:
            df[col] = scale_y.inverse_transform(df[col].values.reshape(-1, 1))

        return df

    def plot_results(
        self,
        df: pd.DataFrame,
        xcol: str,
        ycol: str = "residual",
        no_cols: int = 3,
        fig_size: tuple = (16, 9),
        ylim: list = [-4, 4],
    ):

        if self.optimum_models_performance_df is not None:
            models_used = list(self.optimum_models_performance_df.models.values)
        else:
            models_used = list(self.base_models_performance_df.models.values)

        no_models = len(models_used)
        if no_models <= no_cols:
            figrow = 1
            figcol = no_models
        else:
            figcol = no_cols
            figrow = (no_models // figcol) + 1

        fig, axes = plt.subplots(figrow, figcol, figsize=fig_size)
        plt.subplots_adjust(hspace=0.25)

        if "residual" in ycol:
            fig.suptitle(f"Residual vs Actual Plot {xcol}")
        elif "pred" in ycol:
            fig.suptitle(f"Predicted vs Actual Plot {xcol}")

        for no, model_name in enumerate(models_used):

            if figrow == 1:
                if figcol == 1:
                    ax_used = axes
                else:
                    ax_used = axes[no]

                sns.scatterplot(
                    data=df, x=f"y_{xcol}", y=f"{ycol}_{model_name}", ax=ax_used
                )

                if "residual" in ycol:
                    sns.lineplot(
                        data=df,
                        x=f"y_{xcol}",
                        y=[0] * len(df),
                        color="k",
                        linewidth=2.5,
                        ax=ax_used,
                    )
                    ax_used.set_ylim(ylim)
                elif "pred" in ycol:
                    sns.lineplot(
                        data=df,
                        x=f"y_{xcol}",
                        y=f"y_{xcol}",
                        color="k",
                        linewidth=2.5,
                        ax=ax_used,
                    )
                    ax_used.set_ylim([0, max(df[f"y_{xcol}"])])

                ax_used.set_title(model_name)
                ax_used.grid(visible=True)

            else:
                col_num = no % figcol
                row_num = no // figcol
                sns.scatterplot(
                    data=df,
                    x=f"y_{xcol}",
                    y=f"{ycol}_{model_name}",
                    ax=axes[row_num, col_num],
                )

                if "residual" in ycol:
                    sns.lineplot(
                        data=df,
                        x=f"y_{xcol}",
                        y=[0] * len(df),
                        color="k",
                        linewidth=2.5,
                        ax=axes[row_num, col_num],
                    )
                    axes[row_num, col_num].set_ylim(ylim)
                elif "pred" in ycol:
                    sns.lineplot(
                        data=df,
                        x=f"y_{xcol}",
                        y=f"y_{xcol}",
                        color="k",
                        linewidth=2.5,
                        ax=axes[row_num, col_num],
                    )
                    axes[row_num, col_num].set_ylim([0, max(df[f"y_{xcol}"])])

                axes[row_num, col_num].set_title(model_name)
                axes[row_num, col_num].grid(visible=True)

    def plot_feature_importance(
            self,
            model_name:str,
            features: list = None,
            plot_type: str = "Permutation",
            return_list_importance: bool = False
    ):
        if not features:
            features = self.optimum_model_details.input

        model_specs = self.optimum_model_details.models_evaluated[model_name]
        if not return_list_importance:
            extract_and_plot_feature_importance(
                features=features,
                model=model_specs["model"],
                X_training=self.optimum_model_details.training_data[0],
                y_training=self.optimum_model_details.training_data[1],
                plot_type=plot_type
            )
        else:
            return extract_and_plot_feature_importance(
                features=features,
                model=model_specs["model"],
                X_training=self.optimum_model_details.training_data[0],
                y_training=self.optimum_model_details.training_data[1],
                return_list_importance=True
            )



def extract_and_plot_feature_importance(
    features: list,
    model: callable,
    X_training: Union[np.array, pd.DataFrame],
    y_training: Union[np.array, pd.DataFrame],
    plot_type: str = "Permutation",
    return_list_importance: bool = False
):

    if plot_type == "Permutation":


        perm_importance = permutation_importance(
            model,
            np.ascontiguousarray(X_training),
            y_training,
            n_repeats=10,
            random_state=42,
        )
        sorted_idx = perm_importance.importances_mean.argsort()
        if not return_list_importance:
            fig = plt.figure(figsize=(12, 6))
            plt.barh(
                range(len(sorted_idx)),
                perm_importance.importances_mean[sorted_idx],
                align="center",
            )
            plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
            plt.title(f"{plot_type} Importance {model.__name__}")
            plt.show()

        else:
            return list(np.array(features)[sorted_idx][::-1])

    elif plot_type == "Feature":
        try:
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            if not return_list_importance:
                fig = plt.figure(figsize=(12, 6))
                plt.barh(
                    range(len(sorted_idx)), feature_importance[sorted_idx], align="center"
                )
                plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
                plt.title(f"{plot_type} Importance {model.__name__}")

            else:
                return list(feature_importance[sorted_idx])
        except AttributeError:
            raise("Model does not have feature_importance, use Permutation importance instead")

    elif plot_type == "both":
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))

        perm_importance = permutation_importance(
            model,
            np.ascontiguousarray(X_training),
            y_training,
            n_repeats=10,
            random_state=42,
        )
        sorted_idx = perm_importance.importances_mean.argsort()

        axes[0].barh(
            range(len(sorted_idx)),
            perm_importance.importances_mean[sorted_idx],
            align="center",
        )
        axes[0].set_yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
        axes[0].set_title(f"Permutation Importance {model.__name__}")

        try:
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)

            axes[1].barh(
                range(len(sorted_idx)),
                feature_importance[sorted_idx],
                align="center",
            )
            axes[1].set_yticks(
                range(len(sorted_idx)), np.array(features)[sorted_idx]
            )
            axes[1].set_title(f"Feature Importance {model.__name__}")
        except AttributeError:
            axes[1].plot()

        plt.show()
