
import logging
import os


log_directory = 'logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configure logging to write to a file in the 'logs' directory
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    filename=os.path.join(log_directory, 'app.log'),  # Log file in 'logs' folder
    filemode='a'  # 'a' means append to the log file
)

logger = logging.getLogger(__name__)