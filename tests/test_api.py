import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def test_financial_data():
    """Test financial data endpoint"""
    endpoint = f"{BASE_URL}/api/data/financial"
    data = {
        "symbol": "AAPL",
        "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "end_date": datetime.now().isoformat()
    }
    
    response = requests.post(endpoint, json=data)
    print("\nFinancial Data Response:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200

def test_sentiment_analysis():
    """Test sentiment analysis endpoint"""
    endpoint = f"{BASE_URL}/api/analysis/sentiment"
    data = {
        "texts": [
            "Apple's revenue increased by 25% this quarter",
            "Stock prices dropped after disappointing earnings",
            "New iPhone launch exceeded market expectations"
        ]
    }
    
    response = requests.post(endpoint, json=data)
    print("\nSentiment Analysis Response:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200

def test_price_prediction():
    """Test price prediction endpoint"""
    endpoint = f"{BASE_URL}/api/prediction/price"
    data = {
        "symbol": "AAPL",
        "steps": 5
    }
    
    response = requests.post(endpoint, json=data)
    print("\nPrice Prediction Response:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200

def test_company_info():
    """Test company info endpoint"""
    endpoint = f"{BASE_URL}/api/company/AAPL"
    response = requests.get(endpoint)
    print("\nCompany Info Response:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200

if __name__ == "__main__":
    print("Starting API tests...")
    
    try:
        test_financial_data()
        test_sentiment_analysis()
        test_price_prediction()
        test_company_info()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nError during testing: {str(e)}") 