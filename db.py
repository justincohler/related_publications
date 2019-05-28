import json
from sqlalchemy import create_engine

def db_connect():
    """Return a SQLAlchemy DB engine.
    
    Returns:
        sqlalchemy.Engine -- a DB engine
    """
    with open('credentials.json') as file:
        credentials = json.load(file)

    DB_USER = credentials["DB_USER"]
    DB_PASS = credentials["DB_PASS"]
    DB_CONN = f"postgresql://{DB_USER}:{DB_PASS}@localhost:5433/postgres"
    return create_engine(DB_CONN)
