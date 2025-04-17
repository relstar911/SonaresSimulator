"""
Database Module for SONARES.

This module provides database functionality for storing and
retrieving simulation data and experimental results.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create database connection
DATABASE_URL = os.environ.get('DATABASE_URL')

# Check if DATABASE_URL exists
if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    Base = declarative_base()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db_available = True
else:
    print("Warning: DATABASE_URL not found. Database functionality will be disabled.")
    # Create placeholder objects to avoid errors
    engine = None
    Base = declarative_base()
    SessionLocal = None
    db_available = False

class MaterialRecord(Base):
    """SQLAlchemy model for storing material data."""
    __tablename__ = "materials"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    resonance_frequency = Column(Float)
    density = Column(Float)
    damping_coefficient = Column(Float)
    elasticity = Column(Float)
    destruction_threshold = Column(Float)
    notes = Column(Text, nullable=True)
    is_custom = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
class SimulationResult(Base):
    """SQLAlchemy model for storing simulation results."""
    __tablename__ = "simulations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    material_name = Column(String)
    frequency = Column(Float)
    source_count = Column(Integer)
    source_arrangement = Column(String)
    medium = Column(String)
    reflection_coefficient = Column(Float)
    max_intensity = Column(Float)
    resonance_factor = Column(Float)
    resonance_hotspots = Column(Integer)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
class ExperimentalData(Base):
    """SQLAlchemy model for storing experimental data."""
    __tablename__ = "experimental_data"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    data_json = Column(Text)  # JSON serialized data
    material_name = Column(String)
    frequency_range_min = Column(Float)
    frequency_range_max = Column(Float)
    data_points = Column(Integer)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create all tables in the database if database is available
if DATABASE_URL and engine:
    Base.metadata.create_all(bind=engine)

class DatabaseHandler:
    """Class to handle database operations for SONARES."""
    
    def __init__(self):
        """Initialize database handler."""
        self.engine = engine
        self.SessionLocal = SessionLocal
        self.db_available = db_available
    
    def get_db_session(self):
        """Create and return a new database session."""
        if not self.db_available or self.SessionLocal is None:
            raise ValueError("Database connection not available")
        return self.SessionLocal()
    
    def add_material(self, material_data):
        """
        Add a new material to the database.
        
        Args:
            material_data (dict): Material properties
        
        Returns:
            MaterialRecord: The created material record
        """
        db = self.get_db_session()
        try:
            material = MaterialRecord(**material_data)
            db.add(material)
            db.commit()
            db.refresh(material)
            return material
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def get_materials(self):
        """
        Get all materials from the database.
        
        Returns:
            list: List of material records
        """
        db = self.get_db_session()
        try:
            materials = db.query(MaterialRecord).all()
            return materials
        finally:
            db.close()
    
    def get_material_by_name(self, name):
        """
        Get a material by name.
        
        Args:
            name (str): Material name
        
        Returns:
            MaterialRecord: Material record or None if not found
        """
        db = self.get_db_session()
        try:
            material = db.query(MaterialRecord).filter(MaterialRecord.name == name).first()
            return material
        finally:
            db.close()
    
    def save_simulation_result(self, result_data):
        """
        Save simulation result to the database.
        
        Args:
            result_data (dict): Simulation result data
        
        Returns:
            SimulationResult: The created simulation record
        """
        db = self.get_db_session()
        try:
            result = SimulationResult(**result_data)
            db.add(result)
            db.commit()
            db.refresh(result)
            return result
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def get_simulation_results(self, limit=20):
        """
        Get recent simulation results.
        
        Args:
            limit (int): Maximum number of records to return
        
        Returns:
            list: List of simulation results
        """
        db = self.get_db_session()
        try:
            results = db.query(SimulationResult).order_by(
                SimulationResult.created_at.desc()
            ).limit(limit).all()
            return results
        finally:
            db.close()
    
    def get_simulations_by_material(self, material_name):
        """
        Get simulation results for a specific material.
        
        Args:
            material_name (str): Material name
        
        Returns:
            list: List of simulation results
        """
        db = self.get_db_session()
        try:
            results = db.query(SimulationResult).filter(
                SimulationResult.material_name == material_name
            ).order_by(SimulationResult.created_at.desc()).all()
            return results
        finally:
            db.close()
    
    def save_experimental_data(self, name, data_df, material_name, notes=None):
        """
        Save experimental data to the database.
        
        Args:
            name (str): Name for this experimental dataset
            data_df (pd.DataFrame): Pandas DataFrame with experimental data
            material_name (str): Material name
            notes (str, optional): Additional notes
        
        Returns:
            ExperimentalData: The created experimental data record
        """
        db = self.get_db_session()
        try:
            # Calculate min and max frequency if 'Frequency (Hz)' column exists
            freq_min = None
            freq_max = None
            data_points = len(data_df)
            
            if 'Frequency (Hz)' in data_df.columns:
                freq_min = float(data_df['Frequency (Hz)'].min())
                freq_max = float(data_df['Frequency (Hz)'].max())
            
            # Convert dataframe to JSON
            data_json = data_df.to_json(orient='records')
            
            exp_data = ExperimentalData(
                name=name,
                data_json=data_json,
                material_name=material_name,
                frequency_range_min=freq_min,
                frequency_range_max=freq_max,
                data_points=data_points,
                notes=notes
            )
            
            db.add(exp_data)
            db.commit()
            db.refresh(exp_data)
            return exp_data
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def get_experimental_data(self, limit=20):
        """
        Get recent experimental data.
        
        Args:
            limit (int): Maximum number of records to return
        
        Returns:
            list: List of experimental data records
        """
        db = self.get_db_session()
        try:
            results = db.query(ExperimentalData).order_by(
                ExperimentalData.created_at.desc()
            ).limit(limit).all()
            return results
        finally:
            db.close()
    
    def get_experimental_data_as_df(self, record_id):
        """
        Get experimental data as a pandas DataFrame.
        
        Args:
            record_id (int): ID of the experimental data record
        
        Returns:
            pd.DataFrame: DataFrame containing the experimental data
        """
        db = self.get_db_session()
        try:
            record = db.query(ExperimentalData).filter(ExperimentalData.id == record_id).first()
            if record and record.data_json:
                return pd.read_json(record.data_json, orient='records')
            return None
        finally:
            db.close()