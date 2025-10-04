from src.models.user import User
from src.models.product import Product
from src.models.connection import DBConnection
from src.models.data_service import DataService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting data model project")

    db = DBConnection()
    
    dataService = DataService(db)
    