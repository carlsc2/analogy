from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

engine = None
db_session = None
Base = declarative_base()
Base.query = None

def init_db(path_to_database):
    global engine, db_session, Base
    engine = create_engine('sqlite:///' + path_to_database,
                           convert_unicode=True)
    db_session = scoped_session(sessionmaker(autocommit=False,
                                             autoflush=False,
                                             expire_on_commit=False,
                                             bind=engine))
    Base.query = db_session.query_property()
    import domainDB.models
    Base.metadata.create_all(bind=engine)
    return db_session
