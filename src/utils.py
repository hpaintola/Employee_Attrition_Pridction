from pyspark.sql import DataFrame
from logger import logger  # Importing logger from the external module



def read_parquet_file(spark, file_path: str) -> DataFrame:
    """
    Reads a parquet file into a Spark DataFrame.
    """
    logger.info(f"Reading parquet file from: {file_path}")
    df = spark.read.format("parquet").load(file_path)
    logger.info(f"Data loaded from {file_path}, number of rows: {df.count()}")
    return df