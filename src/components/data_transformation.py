
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from config.config import storage_path
from loggers import logger

class EmployeeDataProcessor:
    def __init__(self, file_path, save_path):
        self.file_path = file_path
        self.save_path = save_path
        self.spark = SparkSession.builder.appName("Employee Data Processing").getOrCreate()

    def load_data(self):
        """
        Load data from the given CSV file.

        :return: Spark DataFrame with the loaded data.
        """
        logger.info("Loading data from %s", self.file_path)
        return self.spark.read.format("csv") \
            .option("header", "true") \
            .option("inferschema", "true") \
            .load(self.file_path)

    def check_null_values(self, df):
        """
        Check for null values in the DataFrame.

        :param df: Spark DataFrame to check for nulls.
        :return: DataFrame with null value counts per column.
        """
        logger.info("Checking for null values")
        return df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])

    def find_mode(self, df, column_name):
        """
        Find the mode of a column in a DataFrame.

        :param df: Spark DataFrame.
        :param column_name: Name of the column.
        :return: Mode value or None if the column is empty.
        """
        logger.info("Finding mode for column: %s", column_name)
        mode_row = (
            df.groupBy(column_name)
            .count()
            .orderBy(desc("count"))
            .limit(1)
            .collect()
        )
        return mode_row[0][0] if mode_row else None

    def handle_null_values(self, df):
        """
        Handle missing values in the DataFrame by imputing or dropping them.

        :param df: Spark DataFrame with missing values.
        :return: DataFrame with missing values handled.
        """
        logger.info("Handling missing values")
        df_dropped = df.dropna(subset=["Employee ID", "Attrition"])
        age_mean = df.select(mean(col("age"))).collect()[0][0]
        monthly_income_mean = df.select(mean(col("Monthly Income"))).collect()[0][0]
        company_tenure_mean = df.select(mean(col("Company Tenure"))).collect()[0][0]

        df_imputed = df_dropped.na.fill({
            "age": age_mean,
            "Monthly Income": monthly_income_mean,
            "Number of Promotions": 0,
            "Number of Dependents": 0,
            "Company Tenure": company_tenure_mean,
            "Gender": "Unknown",
            "Job Role": "Unknown",
            "Work-Life Balance": self.find_mode(df, "Work-Life Balance"),
            "Job Satisfaction": self.find_mode(df, "Job Satisfaction"),
            "Overtime": "No",
            "Education Level": self.find_mode(df, "Education Level"),
            "Marital Status": "Unknown",
            "Job Level": self.find_mode(df, "Job Level"),
            "Company Size": "Unknown",
            "Remote Work": "Unknown",
            "Leadership Opportunities": self.find_mode(df, "Leadership Opportunities"),
            "Innovation Opportunities": self.find_mode(df, "Innovation Opportunities"),
            "Company Reputation": self.find_mode(df, "Company Reputation"),
            "Employee Recognition": self.find_mode(df, "Employee Recognition")
        })

        return df_imputed

    def remove_outliers(self, df, columns):
        """
        Remove outliers from specified columns using the IQR method.

        :param df: Spark DataFrame.
        :param columns: List of column names to process.
        :return: DataFrame with outliers handled.
        """
        logger.info("Removing outliers from columns: %s", columns)
        for column in columns:
            Q1 = df.approxQuantile(column, [0.25], 0.01)[0]
            Q3 = df.approxQuantile(column, [0.75], 0.01)[0]
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df = df.withColumn(
                column,
                when(col(column) < lower_bound, lower_bound)
                .when(col(column) > upper_bound, upper_bound)
                .otherwise(col(column))
            )
        return df

    def remove_duplicates(self, df):
        """
        Remove duplicate rows from the DataFrame.

        :param df: Spark DataFrame.
        :return: DataFrame without duplicates.
        """
        logger.info("Removing duplicate rows")
        return df.dropDuplicates()

    def save_to_parquet(self, df):
        """
        Save the DataFrame to a Parquet file.

        :param df: Spark DataFrame.
        """
        logger.info("Saving data to Parquet format at %s", self.save_path)
        df.write.format("parquet").mode("overwrite").save(self.save_path)

    def process_data(self):
        """
        Main function to process employee data.
        """
        logger.info("Starting data processing pipeline")
        raw_emp_df = self.load_data()

        logger.info("Handling missing values")
        df_imputed = self.handle_null_values(raw_emp_df)

        logger.info("Removing outliers")
        numeric_columns = ['Age', 'Years at Company', 'Number of Promotions', 'Distance from Home', 'Number of Dependents', 'Company Tenure']
        df_no_outliers = self.remove_outliers(df_imputed, numeric_columns)

        logger.info("Removing duplicates")
        df_no_duplicates = self.remove_duplicates(df_no_outliers)

        logger.info("Saving processed data")
        self.save_to_parquet(df_no_duplicates)

# Execute the pipeline
if __name__ == "__main__":
    input_file_path = storage_path
    output_file_path = "C:\\Users\\acer\\100-days-of-machine-learning-main\\iDataMinds\\DE\\silver_emp_df.parquet"
    processor = EmployeeDataProcessor(input_file_path, output_file_path)
    processor.process_data()
