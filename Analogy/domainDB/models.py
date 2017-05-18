from sqlalchemy import Column, Integer, String, UniqueConstraint, ForeignKey, Table, Boolean, Float, Text
from sqlalchemy.orm import relationship, backref
from .database import Base, db_session, engine

class Domain(Base):
    __tablename__ = 'domains'
    id = Column(Integer,
                primary_key=True)
    filepath = Column(String(256),
                      unique=True,
                      nullable=False)
    details = Column(Text, default="{}")#JSON metadata about domain
                      

class Concept(Base):
    __tablename__ = 'concepts'
    id = Column(Integer,
                primary_key=True)
    name = Column(String(256),
                  nullable=False)
    domain = Column(Integer,
                    ForeignKey(Domain.id,
                               ondelete='CASCADE'),
                    nullable=False)
    __table_args__ = (UniqueConstraint('name',
                                       'domain',
                                       name='_cid_uc'),
                      )#allow same concept only in different domains

class Unknown(Base):
    __tablename__ = 'unknowns'
    id = Column(Integer,
                primary_key=True)
    name = Column(String(256),
                  nullable=False)
