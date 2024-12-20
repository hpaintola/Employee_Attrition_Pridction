from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from loggers import logger  
from utils import read_parquet_file

class ModelTrainingPipeline:
    def __init__(self, spark, train_file_path: str, test_file_path: str):
        """
        Initializes the ModelTrainingPipeline class with the provided Spark session and file paths.
        """
        self.spark = spark
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.logger = logger  # Using the imported logger
        self.logger.info("ModelTrainingPipeline initialized")



    def train_logistic_regression(self, train: DataFrame, features_col: str, label_col: str) -> LogisticRegression:
        """
        Trains a Logistic Regression model.
        """
        self.logger.info(f"Training Logistic Regression model with features: {features_col} and label: {label_col}")
        lr = LogisticRegression(featuresCol=features_col, labelCol=label_col)
        model = lr.fit(train)
        self.logger.info("Logistic Regression model trained successfully")
        return model

    def evaluate_model(self, predictions: DataFrame, label_col: str, prediction_col: str):
        """
        Evaluates model predictions using BinaryClassificationEvaluator and MulticlassClassificationEvaluator.
        """
        self.logger.info(f"Evaluating model with label: {label_col} and prediction: {prediction_col}")
        
        # Evaluate AUC
        binary_evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="prediction", metricName="areaUnderROC")
        auc = binary_evaluator.evaluate(predictions)
        self.logger.info(f"Area Under ROC (AUC): {auc}")

        # Evaluate Accuracy
        multi_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col)
        accuracy = multi_evaluator.evaluate(predictions)
        self.logger.info(f"Accuracy: {accuracy}")

    def train_random_forest(self, train: DataFrame, features_col: str, label_col: str) -> RandomForestClassifier:
        """
        Trains a Random Forest model.
        """
        self.logger.info(f"Training Random Forest model with features: {features_col} and label: {label_col}")
        rf = RandomForestClassifier(featuresCol=features_col, labelCol=label_col)
        model = rf.fit(train)
        self.logger.info("Random Forest model trained successfully")
        return model

    def get_feature_importance(self, rf_model, feature_columns: list) -> DataFrame:
        """
        Extracts feature importances from the trained Random Forest model and returns them as a Spark DataFrame.
        """
        self.logger.info("Extracting feature importance from Random Forest model")
        importances = rf_model.featureImportances.toArray()
        importances_list = [(name, float(importance)) for name, importance in zip(feature_columns, importances)]
        
        schema = StructType([
            StructField("Feature", StringType(), True),
            StructField("Importance", FloatType(), True)
        ])
        
        feature_importance_df = self.spark.createDataFrame(importances_list, schema=schema)
        self.logger.info(f"Feature importance extracted, total features: {len(importances_list)}")
        return feature_importance_df

    def run_pipeline(self):
        """
        Runs the entire model training and evaluation pipeline.
        """
        self.logger.info("Starting pipeline execution")

        # Read train and test datasets
        self.logger.info("Reading train and test datasets")
        train = read_parquet_file(self.train_file_path)
        test = read_parquet_file(self.test_file_path)

        # Logistic Regression training and evaluation
        self.logger.info("Starting Logistic Regression training")
        lr_model = self.train_logistic_regression(train, features_col="features", label_col="Attrition_index")
        predictions = lr_model.transform(test)
        predictions.select("features", "Attrition", "Attrition_index", "prediction", "probability").show(5)

        self.evaluate_model(predictions, label_col="Attrition_index", prediction_col="prediction")

        # Random Forest training and feature importance extraction
        self.logger.info("Starting Random Forest training")
        rf_model = self.train_random_forest(train, features_col="features", label_col="Attrition_index")
        
        feature_columns = [
            'scaled_numerical_feature_vector',
            'Work-Life Balance_index',
            'Job Satisfaction_index',
            'Performance Rating_index',
            'Education Level_index',
            'Job Level_index',
            'Company Size_index',
            'Company Reputation_index',
            'Employee Recognition_index',
            'Gender_onehot',
            'Job Role_onehot',
            'Overtime_onehot',
            'Marital Status_onehot',
            'Remote Work_onehot',
            'Leadership Opportunities_onehot',
            'Innovation Opportunities_onehot'
        ]
        
        feature_importance_df = self.get_feature_importance(rf_model, feature_columns)
        feature_importance_df.orderBy("Importance", ascending=False).show()

# Run pipeline
if __name__ == "__main__":

    train_path = "C:\\Users\\acer\\100-days-of-machine-learning-main\\iDataMinds\\DE\\gold_train.parquet"
    test_path = "C:\\Users\\acer\\100-days-of-machine-learning-main\\iDataMinds\\DE\\gold_test.parquet"
    
    pipeline = ModelTrainingPipeline(spark, train_file_path=train_path, test_file_path=test_path)
    pipeline.run_pipeline()