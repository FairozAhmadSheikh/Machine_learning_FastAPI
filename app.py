from pydantic import BaseModel,Field
import pandas as pd

class InputSchema(BaseModel):
    SerialNo:int
    grescore:int
    toefl:int
    university:int
    sop:float
    lor:float
    cgpa:float
    research:int


from fastapi import FastAPI

app=FastAPI()


@app.get("/")
def home():
    return "RUNNING"


import joblib
model=joblib.load("./linear_reg_model.pkl")

@app.post("/data")
def get_pred(body: InputSchema):

    data = body.model_dump()

    # Convert input into DataFrame
    df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(df)

    # Return prediction
    return {
        "prediction": float(prediction[0])
    }
