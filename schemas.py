from pydantic import BaseModel

class IrisInput(BaseModel):
    sepal_width: float
    sepal_length: float
    petal_width: float
    petal_length: float
    
class IrisOutput(BaseModel):
    predicted_class: str
    confidence: float