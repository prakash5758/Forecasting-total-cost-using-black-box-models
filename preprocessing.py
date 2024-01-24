from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


class PreProcessing:
    def __init__(self, data: pd.DataFrame, processing_details: dict = {}) -> None:
        self.df = data.copy(deep=True)
        self.processing_details = processing_details
        self.original_data = data.copy(deep=True)

    def run(
        self,
        filtering_dict: dict,
    ):

        # Calculate variables and drop redundant variables
        self.calculate_variables()

        # drop rows containing nan
        self.df = self.df.dropna().reset_index(drop=True)

        # if cols_to_keep is available, remove unnecessary columns.
        if "cols_to_keep" in filtering_dict.keys():
            self.df = self.filter_columns(filtering_dict.get("cols_to_keep"))

        # Remove statistical outliers based on Median absolute deviation method or z-score.
        if "statistical_filtering" in filtering_dict.keys():

            statistical_filtering_config = filtering_dict.get("statistical_filtering")

            cols_to_filter = statistical_filtering_config.get("cols_to_filter", None)
            threshold = statistical_filtering_config.get("threshold", 3.0)
            type = statistical_filtering_config.get("type", "mad")

            self.df, _ = self.filter_statistics_dataframe(
                type, threshold, cols_to_filter
            )

        # Filter data based on the techincal filters specified.
        if "technical_filtering" in filtering_dict.keys():
            technical_filtering_config = filtering_dict.get("technical_filtering")
            threshold_type ="values"
            if technical_filtering_config.get("percentile", False):
                threshold_type = "percentile"

            for col_name, thresholds in technical_filtering_config["thresholds"].items():
                lower, upper = thresholds

                self.df = self.filter_thresholds_dataframe(col_name, upper, lower, threshold_type)

        # Filter ratio of the data ex round(drillcost / totalcost), 3) == 0.325
        if "ratio_filtering" in filtering_dict.keys():
            ratio_filtering_config = filtering_dict.get("ratio_filtering")

            for col_names, threshold in ratio_filtering_config.items():
                col_names = col_names.split(" / ")

                if len(col_names) == 1:
                    col_names = col_names.split("/")

                if len(col_names) != 2:
                    raise ValueError(f"Unable to split {col_names} ")

                target_col, ratio_col = col_names
                threshold = threshold

                self.df = self.filter_by_ratio(target_col, ratio_col, threshold)

        return self.df.dropna().reset_index(drop=True)

    def calculate_variables(self):
        """
        This function calculates median Latitude & longitude, completion year, and cost/ft of lateral length.
        Then, this function drops redundant columns
        """
        self.df["LatitudeMP"] = self.df.apply(
            lambda x: np.nanmean([x["LatitudeBHWGS84"], x["LatitudeSHWGS84"]]), axis=1
        )
        self.df["LongitudeMP"] = self.df.apply(
            lambda x: np.nanmean([x["LongitudeBHWGS84"], x["LongitudeSHWGS84"]]), axis=1
        )

        self.df["CompletionYear"] = pd.to_datetime(
            self.df["CompletionDate"]
        ).dt.year.astype(int)
        self.df["cost/ft_Lateral"] = (
            10**6 * self.df["TotalWellCost_USDMM"] / self.df["LateralLength_FT"]
        )

        self.df.drop(
            [
                "LatitudeBHWGS84",
                "LatitudeSHWGS84",
                "LongitudeBHWGS84",
                "LongitudeSHWGS84",
                "CompletionDate",
            ],
            axis=1,
            inplace=True,
        )

    def filter_columns(self, cols_to_keep: list) -> pd.DataFrame:
        """
        This function updates the self.df dataframe to keep the columns specified in cols_to_keep list.
        """
        if set(cols_to_keep) - set(self.df.columns):
            raise KeyError(
                f"Columns {set(cols_to_keep) - set(self.df.columns)} not available in the dataset"
            )

        df_filtered = self.df[cols_to_keep]

        return df_filtered

    def filter_statistics_dataframe(
        self, type: str = "mad", threshold: float = 3.0, numeric_cols: list = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This function filters a given dataframe based on median absolute deviation or z-score methods.
        """

        self.processing_details["statistical_filtering"] = {
            "type": type,
            "threshold": threshold,
            "columns_filtered": numeric_cols,
        }

        if not numeric_cols:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if type == "mad":
            mad = self.df[numeric_cols].apply(stats.median_abs_deviation)
            score = (
                self.df[numeric_cols] - self.df[numeric_cols].median()
            ).abs() <= threshold * mad
        elif type == "z-score":
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            z_score = abs(self.df[numeric_cols].apply(stats.zscore))
            score = z_score <= threshold

        cols_skipped = list(set(self.df.columns) - set(numeric_cols))
        df_filtered = self.df[score]

        for col in cols_skipped:
            df_filtered[col] = self.df[col]

        df_outliers = self.df[~score]

        return df_filtered, df_outliers

    def filter_thresholds_dataframe(
        self,
        col_name: str,
        upper_threshold: float = None,
        lower_threshold: float = None,
        types_of_threshold: str = "value"
    ) -> pd.DataFrame:
        """
        This function removes all data less than the lower threshold and more than upper
        threshold from the given column name.
        """
        if col_name not in self.df.columns:
            raise AttributeError(
                f"{col_name} not in data columns. Available columns = {list(self.df.columns)}"
            )

        if types_of_threshold == "percentile":
            upper_threshold = int(self.df[col_name].quantile(upper_threshold))
            lower_threshold = int(self.df[col_name].quantile(lower_threshold))

        if "technical_filtering" in self.processing_details.keys():
            self.processing_details["technical_filtering"].update(
                {col_name: (lower_threshold, upper_threshold)}
            )
        else:
            self.processing_details["technical_filtering"] = {
                col_name: (lower_threshold, upper_threshold)
            }

        df_filtered = self.df
        if lower_threshold:
            df_filtered = df_filtered[(df_filtered[col_name] > lower_threshold)]

        if upper_threshold:
            df_filtered = df_filtered[(df_filtered[col_name] <= upper_threshold)]

        return df_filtered

    def filter_by_ratio(
        self, target_col_name: str, ratio_col_name: str, threshold: float
    ) -> pd.DataFrame:
        """
        This function removes all data less than the lower threshold and more than upper
        threshold from the given column name.
        """

        if target_col_name not in self.df.columns:
            raise KeyError(
                f"{target_col_name} not in data columns. Available columns = {list(self.df.columns)}"
            )

        if ratio_col_name not in self.df.columns:
            raise KeyError(
                f"{ratio_col_name} not in data columns. Available columns = {list(self.df.columns)}"
            )

        if "technical_filtering" in self.processing_details.keys():
            self.processing_details["technical_filtering"].update(
                {f"{target_col_name} / {ratio_col_name}": threshold}
            )
        else:
            self.processing_details["technical_filtering"] = {
                f"{target_col_name} / {ratio_col_name}": threshold
            }

        df_filtered = self.df[
            round((self.df[target_col_name] / self.df[ratio_col_name]), 3) == threshold
        ]

        return df_filtered
