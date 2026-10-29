from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# Allow the frontend (index.html) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your site's URL after deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("linearmodel.pkl")


class datascience(BaseModel):
    AdvertisingBudget: float


@app.get("/")
def test():
    return {"msg": "uvicorn is running"}


@app.post("/predict")
def datainsert(ds: datascience):
    sales = model.predict([[ds.AdvertisingBudget]])
    return {"predictedsales": round(float(sales[0][0]), 2)}