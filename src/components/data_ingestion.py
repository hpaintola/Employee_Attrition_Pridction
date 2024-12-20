import pandas as pd
from io import BytesIO
from zipfile import ZipFile
import requests
from config.config import Api_url, auth_key, storage_path
from loggers import logger





class DataIngestion:
    def __init__(self, url, headers, save_path, file_name):
        """
        Initialize the DataIngestion class.

        :param url: URL to download the data.
        :param headers: Headers to include in the HTTP request.
        :param save_path: Path to save the extracted CSV file.
        :param file_name: Name of the file to extract from the zip.
        """
        self.url = url 
        self.headers = headers
        self.save_path = save_path
        self.file_name = file_name

    def initiate_Data_ingestion(self):
        """
        Fetch the zip file from the URL, extract the specified CSV file, and save it.
        """
        try:
            response = requests.get(self.url,headers=self.headers, stream=True)
            response.raise_for_status()
        

            # Extract and read the specific CSV file from the zip directly into a DataFrame
            with ZipFile(BytesIO(response.content)) as z:
                if self.file_name in z.namelist():
                    with z.open(self.file_name) as f:
                        df = pd.read_csv(f)
                        print("File loaded successfully")
                else:
                    raise FileNotFoundError(f"{self.file_name} not found in the downloaded zip file")
                

            # Save the DataFrame to CSV

            df.to_csv(self.save_path, index=False)
            logger.info(f"Data saved to {self.save_path} successfully!")

        # Handle specific errors
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} [status code: {response.status_code if 'response' in locals() else 'unknown'}]")
        except FileNotFoundError as fnf_err:
            print(f"File error: {fnf_err}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Replace with appropriate values
    url = Api_url
    headers = {"Authorization": auth_key}
    save_path = storage_path
    file_name = "train.csv"

    data_ingestion = DataIngestion(url, headers, save_path, file_name)
    data_ingestion.initiate_Data_ingestion()