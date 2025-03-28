# QuantBrain

A Python-based AI tool for quantitative finance that combines market prediction, sentiment analysis, and backtesting capabilities. Built with FastAPI and Streamlit.

## Features

- 📈 Real-time financial data retrieval and visualization
- 🤖 AI-powered sentiment analysis of financial news and social media
- 🔮 Price prediction using LSTM neural networks
- 📊 Interactive charts and data visualization
- 🏢 Comprehensive company information
- 🚀 FastAPI backend with RESTful endpoints
- 💻 Modern Streamlit web interface

## Project Structure

```
quantbrain/
├── api/              # FastAPI application
│   └── main.py      # API endpoints
├── src/             # Core source code
│   ├── data/       # Data retrieval and processing
│   ├── analysis/   # Analysis tools
│   └── models/     # AI models
├── tests/          # Test suite
├── streamlit_app.py # Streamlit web interface
└── requirements.txt # Project dependencies
```

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantbrain.git
cd quantbrain
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the FastAPI backend server:
```bash
uvicorn api.main:app --reload
```

2. In a new terminal, start the Streamlit frontend:
```bash
streamlit run streamlit_app.py
```

The application will be available at:
- Frontend: http://localhost:8501
- API Documentation: http://localhost:8000/docs

## API Endpoints

- `POST /api/data/financial`: Get financial data for a stock
- `POST /api/analysis/sentiment`: Analyze sentiment of financial texts
- `POST /api/prediction/price`: Get price predictions
- `GET /api/company/{symbol}`: Get company information

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```
3. Make your changes
4. Run the tests:
```bash
pytest tests/
```
5. Commit your changes:
```bash
git commit -m "Add your feature"
```
6. Push to your branch:
```bash
git push origin feature/your-feature-name
```
7. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FastAPI for the backend framework
- Streamlit for the web interface
- Hugging Face for the sentiment analysis model
- Yahoo Finance for financial data
- PyTorch for the LSTM model

## Project Status

QuantBrain is under active development. Expect breaking changes and unfinished components. Contributions are welcome, but features may evolve rapidly.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
