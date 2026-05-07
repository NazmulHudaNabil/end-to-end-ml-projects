from sqlalchemy import Boolean, Column, Float, Integer, String
from database import Base

class Car(Base):
    __tablename__ = "cars"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    Levy = Column(Float, nullable=False)
    Manufacturer = Column(String(50), nullable=False)
    Prod_year = Column(Integer, nullable=False)
    Category = Column(String(50), nullable=False)
    Leather_interior = Column(String(10), nullable=False)
    Fuel_type = Column(String(20), nullable=False)
    Gear_box_type = Column(String(20), nullable=False)
    Drive_wheels = Column(String(20), nullable=False)
    Engine_volume = Column(String(20), nullable=False)
    Cylinders = Column(Integer, nullable=False)
    Airbags = Column(Integer, nullable=False)
    Doors = Column(Integer, nullable=False)
    Wheel_position = Column(String(10), nullable=False)
    Color = Column(String(20), nullable=False)
    predicted_price = Column(Float, nullable=True) 