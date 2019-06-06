import scholarly
import time
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from db import db_connect
import random

Base = declarative_base()

class Paper(Base):

    __tablename__ = "paper"
    
    id = Column(Integer, primary_key=True)
    title = Column(String, unique=True)
    abstract = Column(String)
    links = Column(Integer)
    search_term = Column(String)

    def __init__(self, title, abstract, links, search_term):
        self.title = title
        self.abstract = abstract
        self.links = int(links)
    

def create_schema(engine):
    """Use the given engine to drop and recreate all models.
    
    Arguments:
        engine {sqlalchemy.Engine} -- The SQL Alchemy engine.
    """
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

def get_papers(session):
    return session.query(Paper).all()

if __name__ == "__main__":
    engine = db_connect()
    # create_schema(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # for search_term in ["Early Childhood Development", "Nature Conservation", "Machine Learning"]:
    search_term = 'Gauss'
    search_query = scholarly.search_pubs_query(search_term)

    count = 10000
    for i, res in enumerate(search_query):
        print(i)
        if count < 0:
            break

        if 'title' not in res.bib or 'abstract' not in res.bib:
            continue

        paper = Paper(res.bib['title'], res.bib['abstract'], res.citedby, search_term)
        try:
            session.add(paper)
            session.commit()
            print(f"Added {res.bib['title']}")
        except Exception as ex:
            print(f"Failed to add {res.bib['title']}")
            print(ex)
        finally:
            session.rollback()

        count -= 1
        # time.sleep(1.5)
        # time.sleep(3.0 + random.random()*3)
    
    session.close()



