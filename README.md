# 📊 Sentiment Analysis for the Stock Market

Interactive and automated tool for analyzing the sentiment of financial assets based on multiple news sources. The system quickly evaluates market sentiment (positive, neutral, or negative) for stocks listed on the Ibovespa, S&P 500, and Nasdaq, using machine learning and natural language processing.

## 🚀 Features

✅ News collection and sentiment analysis from multiple sources:
- Yahoo Finance
- Google News
- Reuters
- MarketWatch
- Investing.com
- Finviz

✅ Automatic sentiment classification:
- Positive
- Neutral
- Negative

✅ Automatic translation to English to ensure compatibility with FinBERT

✅ Interactive Jupyter Notebook interface:
- 🌎 Select market universe (Ibovespa, S&P 500, Nasdaq)
- 📰 Choose news sources
- 🏆 Define Top K assets by sentiment
- 💾 Export results as `.csv`

---

## ⚙️ Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/erikmilesi-data/fin-sentiment-analysis.git
cd fin-sentiment-analysis
pip install -r requirements.txt
