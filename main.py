from fastapi import FastAPI
from pydantic  import BaseModel
import joblib

app=FastAPI()
model=joblib.load("linearmodel.pkl")

class datascience(BaseModel):
    AdvertisingBudget:float

@app.get("/")
def test():
    return{"msg":"uvicorn is running"}

@app.post("/predict")
def datainsert(ds:datascience):
    sales=model.predict([[ds.AdvertisingBudget]])
    return {"predictedsales":round(float(sales[0][0]),2)}