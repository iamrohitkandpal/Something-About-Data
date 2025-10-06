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
    
    try:
        user = User(name="John Doe", email="john@example.com", age=30)
        product = Product(name="Sample Product", price=29.99, category="Electronics")
        
        logger.info(f"Created user: {user}")
        logger.info(f"Created product: {product}")
   
    except Exception as e:
        logger.error(f"Error in main: {e}")
        
    finally:
        db.close()
        
if __name__ == "__main__":
    main()