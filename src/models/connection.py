import sqlite3
from pathlib import Path


class DBConnection:

    def __init__(self, db_path="data/database.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLES IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT,
                age INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLES IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                price REAL,
                category TXT
            )
        """)
        self.connection.commit()
        
    def close(self):
        if self.connection:
            self.connection.close()        
            
class DataService:
    """Handles saving and loading data using the database."""
    
    def __init__(self, db_connection):
        self.db = db_connection  # The DB connection object
    
    def save_user(self, user):
        """Save a user to the database."""
        cursor = self.db.connection.cursor()
        cursor.execute("INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
                       (user.name, user.email, user.age))
        self.db.connection.commit()
    
    def save_product(self, product):
        """Save a product to the database."""
        cursor = self.db.connection.cursor()
        cursor.execute("INSERT INTO products (name, price, category) VALUES (?, ?, ?)",
                       (product.name, product.price, product.category))
        self.db.connection.commit()