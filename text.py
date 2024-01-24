{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyspark.sql import SparkSession\n",
    "\n",
    "#enableHiveSupport() -> enables sparkSession to connect with Hive\n",
    "warehouse_location = abspath('spark-warehouse')\n",
    "spark = SparkSession \\\n",
    "    .builder \\\n",
    "    .appName(\"SparkByExamples.com\") \\\n",
    "    .config(\"spark.sql.warehouse.dir\", warehouse_location) \\\n",
    "    .enableHiveSupport() \\\n",
    "    .getOrCreate()"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
