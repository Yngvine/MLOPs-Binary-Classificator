"""
API main module using FastAPI for mylib functions endpoints.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import uvicorn
import sys
import os


from custom_lib import predict

app = FastAPI(
    title="Rice Classification API",
    description="API endpoints for rice grain classification.",
    version="0.1.0",
)

templates = Jinja2Templates(directory="templates")

class RiceFeatures(BaseModel):
    Area: int
    MajorAxisLength: float
    MinorAxisLength: float
    Eccentricity: float
    ConvexArea: int
    EquivDiameter: float
    Extent: float
    Perimeter: float
    Roundness: float
    AspectRation: float

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Pass the 'request' object to the template so it can access app.routes
    return templates.TemplateResponse(request=request, name="home.html")

@app.post("/classify/")
async def classify_rice(features: RiceFeatures):
    """Classify rice grain based on features."""
    try:
        # Convert features to list in correct order
        feature_list = [
            features.Area,
            features.MajorAxisLength,
            features.MinorAxisLength,
            features.Eccentricity,
            features.ConvexArea,
            features.EquivDiameter,
            features.Extent,
            features.Perimeter,
            features.Roundness,
            features.AspectRation
        ]
        
        pred_class = predict(feature_list)
        return {"predicted_class": pred_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api.api_main:app", host="0.0.0.0", port=8000, reload=True)