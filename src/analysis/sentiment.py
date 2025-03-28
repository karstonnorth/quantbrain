from transformers import pipeline
from typing import List, Dict, Union
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline(
            "sentiment-analysis",
            model="microsoft/phi-2",  # Using a smaller, more stable model
            device=-1  # Use CPU by default
        )
        
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            result = self.model(text)[0]
            return {
                'label': result['label'],
                'score': result['score'],
                'text': text
            }
        except Exception as e:
            print(f"Error analyzing text: {str(e)}")
            return {'error': str(e)}

    def analyze_texts(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment analysis results
        """
        return [self.analyze_text(text) for text in texts]

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str
    ) -> pd.DataFrame:
        """
        Analyze sentiment for a column in a DataFrame.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of the column containing text
            
        Returns:
            DataFrame with added sentiment columns
        """
        if text_column not in df.columns:
            raise ValueError(f"Column {text_column} not found in DataFrame")
            
        results = self.analyze_texts(df[text_column].tolist())
        
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
        
        return df 