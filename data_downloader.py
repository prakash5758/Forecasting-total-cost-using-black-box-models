from typing import Tuple

import pandas as pd


class Downloader:
    def __init__(
        self,
        dataset_enverus: str,
        basin_of_interest: str,
        reservoirs_of_interest: Tuple[str],
        dataset_completions: str = None,
        impute_from_enverus: bool = True,
    ):

        self.dataset_enverus = dataset_enverus
        self.dataset_completions = dataset_completions
        self.basin_of_interest = basin_of_interest
        self.reservoirs_of_interest = reservoirs_of_interest
        self.impute = impute_from_enverus

    def run(self, spark) -> pd.DataFrame:
        data_private = self.download_data_completions(spark)
        data_private.rename(columns={"TotalWellCost_ENV_USDMM":"TotalWellCost_USDMM",
                                     "DrillCost_ENV_USDMM":"DrillCost_USDMM",
                                    "FacilitiesCost_ENV_USDMM":"FacilitiesCost_USDMM",
                                    "CompletionCost_ENV_USDMM":"CompletionCost_USDMM"}, inplace=True)
        
        if self.impute:
            data_enverus = self.download_data_enverus(spark)

            drilling_outliers = (
                round(
                    data_private["DrillCost_USDMM"] / data_private["TotalWellCost_USDMM"], 3,
                ) != 0.325
            )
            facilities_outliers = (
                round(
                    data_private["FacilitiesCost_USDMM"] / data_private["TotalWellCost_USDMM"], 3,
                ) != 0.05
            )
            completions_outliers = (
                round(
                    data_private["CompletionCost_USDMM"] / data_private["TotalWellCost_USDMM"], 3,
                ) != 0.625
            )

            data_private["cost_filter"] = (
                drilling_outliers | facilities_outliers | completions_outliers
            )
            data_private[data_private["cost_filter"]]

            for index, row in data_private.iterrows():

                if row["cost_filter"]:
                    row_from_enverus = data_enverus[
                        data_enverus["API14"] == row["API14"]
                    ]
                    if row_from_enverus.shape[0]>0:
                        data_private.loc[index, "DrillCost_USDMM"] = float(
                            row_from_enverus["DrillCost_USDMM"].values
                        )
                        data_private.loc[index, "FacilitiesCost_USDMM"] = float(
                            row_from_enverus["FacilitiesCost_USDMM"].values
                        )
                        data_private.loc[index, "CompletionCost_USDMM"] = float(
                            row_from_enverus["CompletionCost_USDMM"].values
                        )
                        data_private.loc[index, "TotalWellCost_USDMM"] = float(
                            row_from_enverus["TotalWellCost_USDMM"].values
                        )

            return data_private

        return data_private

    def download_data_enverus(self, spark) -> pd.DataFrame:

        dataset_sql_table = spark.table(self.dataset_enverus)
        dataset_sql_table.createOrReplaceTempView("dataset_table")

        data_df = spark.sql(
            f"""select API10, API14, WellName, CompletionCost_USDMM, DrillCost_USDMM,
            FacilitiesCost_USDMM, TotalWellCost_USDMM
      from dataset_table
      where CompletionDate>'2014-01-01'
        and HoleDirection='HORIZONTAL'
        and TotalWellCost_USDMM is not null
        and BasinQuantum='{self.basin_of_interest}'"""
        ).toPandas()

        return data_df

    def download_data_completions(self, spark) -> pd.DataFrame:

        dataset_sql_table = spark.table(self.dataset_completions)
        dataset_sql_table.createOrReplaceTempView("dataset_table")

        data_df = spark.sql(
            f"""select API10, API14, WellName, LatitudeBHWGS84, LatitudeSHWGS84, LongitudeBHWGS84, LongitudeSHWGS84,
            CompletionDate, CompletionCost_ENV_USDMM, DrillCost_ENV_USDMM, FacilitiesCost_ENV_USDMM,
            TotalWellCost_ENV_USDMM, Proppant_LBSPerFT, Fluid_BBLPerFT, TVD_FT, LateralLength_FT, OperatorGold, ReservoirGoldConsolidated
            from dataset_table
            where CompletionDate>'2014-01-01'
            and HoleDirection='H'
            and TotalWellCost_ENV_USDMM is not null
            and BasinQuantum='{self.basin_of_interest}'
            and ReservoirGoldConsolidated in {self.reservoirs_of_interest}"""
        ).toPandas()

        return data_df
    