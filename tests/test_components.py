import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.data.financial import FinancialData
from src.analysis.sentiment import SentimentAnalyzer
from src.models.price_predictor import PricePredictor

@pytest.fixture
def financial_data():
    return FinancialData()

@pytest.fixture
def sentiment_analyzer():
    return SentimentAnalyzer()

@pytest.fixture
def price_predictor():
    return PricePredictor()

def test_get_stock_data(financial_data):
    # Test getting data for a well-known stock
    data = financial_data.get_stock_data('AAPL', period='1mo')
    assert not data.empty
    assert 'Close' in data.columns
    assert 'Open' in data.columns
    assert 'High' in data.columns
    assert 'Low' in data.columns
    assert 'Volume' in data.columns

def test_sentiment_analysis(sentiment_analyzer):
    # Test sentiment analysis with some sample financial texts
    texts = [
        "The company's revenue increased by 25% this quarter",
        "Stock prices plummeted after disappointing earnings",
        "New product launch exceeded market expectations"
    ]
    results = sentiment_analyzer.analyze_texts(texts)
    assert len(results) == len(texts)
    assert all('label' in result for result in results)
    assert all('score' in result for result in results)

def test_price_prediction(price_predictor, financial_data):
    # Get some sample data
    data = financial_data.get_stock_data('AAPL', period='1mo')
    if not data.empty:
        # Test prediction
        price_predictor.train(data['Close'])
        predictions = price_predictor.predict(data['Close'], steps=5)
        assert len(predictions) == 5
        assert all(isinstance(pred, (int, float)) for pred in predictions)

def test_company_info(financial_data):
    # Test getting company information
    info = financial_data.get_company_info('AAPL')
    assert isinstance(info, dict)
    assert 'name' in info
    assert 'sector' in info
    assert 'industry' in info 