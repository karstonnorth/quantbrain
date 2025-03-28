from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd

from src.data.financial import FinancialData
from src.analysis.sentiment import SentimentAnalyzer
from src.models.price_predictor import PricePredictor

app = FastAPI(
    title="QuantBrain API",
    description="AI-powered quantitative finance analysis tool",
    version="1.0.0"
)

# Initialize components
financial_data = FinancialData()
sentiment_analyzer = SentimentAnalyzer()
price_predictor = PricePredictor()

class StockRequest(BaseModel):
    symbol: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class SentimentRequest(BaseModel):
    texts: List[str]

class PredictionRequest(BaseModel):
    symbol: str
    steps: int = 5

@app.get("/")
async def root():
    return {"message": "Welcome to QuantBrain API"}

@app.post("/api/data/financial")
async def get_financial_data(request: StockRequest):
    try:
        data = financial_data.get_stock_data(
            request.symbol,
            request.start_date,
            request.end_date
        )
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found")
        return data.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    try:
        results = sentiment_analyzer.analyze_texts(request.texts)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prediction/price")
async def predict_price(request: PredictionRequest):
    try:
        # Get historical data
        data = financial_data.get_stock_data(request.symbol)
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Train model and make predictions
        price_predictor.train(data['Close'])
        predictions = price_predictor.predict(data['Close'], request.steps)
        
        return {
            "symbol": request.symbol,
            "predictions": predictions.tolist(),
            "last_date": data.index[-1].strftime("%Y-%m-%d")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/company/{symbol}")
async def get_company_info(symbol: str):
    try:
        info = financial_data.get_company_info(symbol)
        if not info:
            raise HTTPException(status_code=404, detail="Company not found")
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 