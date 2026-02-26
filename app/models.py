from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class DataRun(Base):
    __tablename__ = "data_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime)
    params = Column(Text)
    csv_path = Column(String)
    rows = Column(Integer, default=0)
    status = Column(String)
    log = Column(Text, default="")

class TrainRun(Base):
    __tablename__ = "train_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime)
    params = Column(Text)
    epochs = Column(Integer)
    status = Column(String)
    log_path = Column(String, default="")
    save_dir = Column(String, default="")

class GenerationRun(Base):
    __tablename__ = "generation_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime)
    train_run_id = Column(Integer, ForeignKey("train_runs.id", ondelete="CASCADE"))
    image_path = Column(String)
    image_filename = Column(String)
    status = Column(String)
    train_run = relationship("TrainRun", backref="generation_runs", passive_deletes=True)