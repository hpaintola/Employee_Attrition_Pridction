from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from loggers import logger
from utils import read_parquet_file

class FeatureEngineeringPipeline:
    def __init__(self, spark: SparkSession):
        self.spark = spark



    def split_data(self, df: DataFrame, train_split: float = 0.7, test_split: float = 0.3):
        """
        Splits the DataFrame into training and testing datasets.
        """
        logger.info(f"Splitting data into training ({train_split*100}%) and testing ({test_split*100}%) datasets.")
        return df.randomSplit([train_split, test_split])

    def assemble_numerical_features(self, train: DataFrame, test: DataFrame, numeric_cols: list) -> (DataFrame, DataFrame):
        """
        Assembles numerical features into a single feature vector and scales them.
        """
        logger.info("Assembling and scaling numerical features.")

        # Assemble numerical features
        numerical_vector_assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numerical_feature_vector")
        train = numerical_vector_assembler.transform(train)
        test = numerical_vector_assembler.transform(test)

        # Scale the numerical features
        scaler = StandardScaler(inputCol="numerical_feature_vector",
                                 outputCol="scaled_numerical_feature_vector",
                                 withStd=True, withMean=True)
        scaler_model = scaler.fit(train)
        train = scaler_model.transform(train)
        test = scaler_model.transform(test)

        return train, test

    def encode_ordinal_features(self, train: DataFrame, test: DataFrame, ordinal_cols: list) -> (DataFrame, DataFrame):
        """
        Encodes ordinal features using StringIndexer.
        """
        logger.info("Encoding ordinal features using StringIndexer.")

        ordinal_index_cols = [col + "_index" for col in ordinal_cols]
        indexer = StringIndexer(inputCols=ordinal_cols, outputCols=ordinal_index_cols)
        indexer_model = indexer.fit(train)
        train = indexer_model.transform(train)
        test = indexer_model.transform(test)

        return train, test

    def encode_nominal_features(self, train: DataFrame, test: DataFrame, nominal_cols: list) -> (DataFrame, DataFrame):
        """
        Encodes nominal features using StringIndexer and OneHotEncoder.
        """
        logger.info("Encoding nominal features using StringIndexer and OneHotEncoder.")

        # Create StringIndexers and OneHotEncoders
        indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in nominal_cols]
        one_hot_encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_onehot") for col in nominal_cols]

        # Combine into a pipeline
        pipeline = Pipeline(stages=indexers + one_hot_encoders)
        pipeline_model = pipeline.fit(train)

        train = pipeline_model.transform(train)
        test = pipeline_model.transform(test)

        return train, test

    def encode_target_column(self, train: DataFrame, test: DataFrame, target_col: str) -> (DataFrame, DataFrame):
        """
        Encodes the target column using StringIndexer.
        """
        logger.info(f"Encoding target column '{target_col}' using StringIndexer.")

        target_indexer = StringIndexer(inputCol=target_col, outputCol=target_col + "_index")
        train = target_indexer.fit(train).transform(train)
        test = target_indexer.fit(test).transform(test)

        return train, test

    def drop_unused_columns(self, df: DataFrame, cols_to_drop: list) -> DataFrame:
        """
        Drops unused columns from the DataFrame.
        """
        logger.info(f"Dropping unused columns: {cols_to_drop}")
        return df.drop(*cols_to_drop)

    def assemble_all_features(self, train: DataFrame, test: DataFrame, feature_cols: list) -> (DataFrame, DataFrame):
        """
        Assembles all feature columns into a single feature vector.
        """
        logger.info("Assembling all feature columns into a single feature vector.")

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        train = assembler.transform(train)
        test = assembler.transform(test)

        return train, test

    def save_parquet_file(self, df: DataFrame, file_path: str):
        """
        Saves a DataFrame to a parquet file.
        """
        logger.info(f"Saving DataFrame to parquet file at path: {file_path}")
        df.write.format("parquet").mode("overwrite").save(file_path)

    def run_pipeline(self, parquet_file_path: str, train_output_path: str, test_output_path: str):
        """
        Complete feature engineering pipeline.
        """
        logger.info("Starting the feature engineering pipeline.")

        # Read data
        df = read_parquet_file(parquet_file_path)

        # Split data into training and testing
        train, test = self.split_data(df)

        # Select numerical columns
        numeric_cols = [col for col, dtype in df.dtypes if dtype in ['int', 'double']]
        numeric_cols.remove('Employee ID')

        # Assemble and scale numerical features
        train, test = self.assemble_numerical_features(train, test, numeric_cols)

        # Encode ordinal features
        ordinal_cols = ["Work-Life Balance", "Job Satisfaction", "Performance Rating", "Education Level",
                        "Job Level", "Company Size", "Company Reputation", "Employee Recognition"]
        train, test = self.encode_ordinal_features(train, test, ordinal_cols)

        # Encode nominal features
        nominal_cols = ["Gender", "Job Role", "Overtime", "Marital Status", "Remote Work",
                        "Leadership Opportunities", "Innovation Opportunities"]
        train, test = self.encode_nominal_features(train, test, nominal_cols)

        # Encode target column
        train, test = self.encode_target_column(train, test, "Attrition")

        # Drop unused columns
        train = self.drop_unused_columns(train, ordinal_cols + nominal_cols)
        test = self.drop_unused_columns(test, ordinal_cols + nominal_cols)

        # Assemble all features
        feature_cols = ["scaled_numerical_feature_vector"] + [col + "_index" for col in ordinal_cols] + \
                       [col + "_onehot" for col in nominal_cols]
        train, test = self.assemble_all_features(train, test, feature_cols)

        # Save to parquet files
        self.save_parquet_file(train, train_output_path)
        self.save_parquet_file(test, test_output_path)

        logger.info("Feature engineering pipeline completed successfully.")
        return train, test


if __name__ == "__main__":
    spark = SparkSession.builder.appName("FeatureEngineeringPipeline").getOrCreate()

    pipeline = FeatureEngineeringPipeline(spark)

    parquet_file_path = "C:\\Users\\acer\\100-days-of-machine-learning-main\\iDataMinds\\DE\\silver_emp_df.parquet"
    train_output_path = "C:\\Users\\acer\\100-days-of-machine-learning-main\\iDataMinds\\DE\\gold_train.parquet"
    test_output_path = "C:\\Users\\acer\\100-days-of-machine-learning-main\\iDataMinds\\DE\\gold_test.parquet"

    pipeline.run_pipeline(parquet_file_path, train_output_path, test_output_path)

