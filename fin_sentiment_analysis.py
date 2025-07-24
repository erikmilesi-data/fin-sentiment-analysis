# Versão atualizada com todos os ajustes solicitados no código de análise de sentimento com FinBERT
import requests
import pandas as pd
import yfinance as yf
import feedparser
import html
import re
import csv
import io
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython.display import display, clear_output
from deep_translator import GoogleTranslator
from transformers import pipeline

# Tradutor
def traduzir_pt_para_en(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print(f"⚠️ Erro na tradução: {e}")
        return text

# FinBERT carregado uma vez
finbert_pipe = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
def _pipe_finbert(text):
    return finbert_pipe(text)

# Universos
def universe_ibov():
    hdr = {"User-Agent": "Mozilla/5.0"}
    url = "https://sistemaswebb3-listados.b3.com.br/listedCompaniesPage/getIbovespaConstituents?language=pt-br"
    try:
        r = requests.get(url, headers=hdr, timeout=15)
        if r.headers.get("Content-Type", "").startswith("application/json"):
            data = r.json()
            return [d["codNegotiation"] + ".SA" for d in data["results"]]
    except Exception:
        pass
    wiki = "https://pt.wikipedia.org/wiki/Lista_de_companhias_citadas_no_Ibovespa"
    soup = BeautifulSoup(requests.get(wiki, headers=hdr, timeout=15).text, "lxml")
    tbl = soup.find("table", {"class": "wikitable"})
    return [tr.find_all("td")[0].text.strip() + ".SA" for tr in tbl.find_all("tr")[1:]]

def universe_sp500():
    w = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    soup = BeautifulSoup(requests.get(w, timeout=30).text, "lxml")
    tbl = soup.find("table", id="constituents")
    return [tr.find_all("td")[0].text.strip().replace(".", "-") for tr in tbl.find_all("tr")[1:]]

def universe_nasdaq():
    url = "https://old.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download"
    rdr = csv.DictReader(io.StringIO(requests.get(url, timeout=30).text))
    return [row["Symbol"] for row in rdr]

UNIVERSE_MAP = {"ibov": universe_ibov, "sp500": universe_sp500, "nasdaq": universe_nasdaq}

# Notícias
def company_name(tkr):
    try:
        info = yf.Ticker(tkr).info
        return info.get("shortName") or info.get("longName") or tkr
    except Exception:
        return tkr

def news_yf(tkr, n=60):
    try:
        items = yf.Ticker(tkr).news or []
    except Exception:
        return []
    hd = []
    for it in items:
        tx = it.get("title") or it.get("headline") or it.get("summary")
        if tx:
            hd.append(" ".join(tx.split()))
        if len(hd) >= n:
            break
    return hd

def news_google(tkr, n=60):
    name = company_name(tkr)
    q = requests.utils.quote(f'"{name}" OR "{tkr}"')
    url = f"https://news.google.com/rss/search?q={q}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    feed = feedparser.parse(url)
    return [html.unescape(re.sub(r"\s+", " ", e.title)) for e in feed.entries[:n]]

def news_reuters(tkr, n=60):
    url = "https://www.reuters.com/rssFeed/businessNews"
    feed = feedparser.parse(url)
    return [html.unescape(re.sub(r"\s+", " ", e.title))
            for e in feed.entries if tkr.split('.')[0].upper() in e.title.upper()][:n]

def news_marketwatch(tkr, n=60):
    url = "https://feeds.marketwatch.com/marketwatch/topstories/"
    feed = feedparser.parse(url)
    return [html.unescape(re.sub(r"\s+", " ", e.title))
            for e in feed.entries if tkr.split('.')[0].upper() in e.title.upper()][:n]

def news_investing(tkr, n=60):
    try:
        hdr = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.investing.com/search/?q={tkr}"
        resp = requests.get(url, headers=hdr, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        return [a.get_text(strip=True) for a in soup.select(".js-inner-all-results-quote-item .title")][:n]
    except Exception:
        return []

def news_finviz(tkr, n=60):
    try:
        hdr = {"User-Agent": "Mozilla/5.0"}
        url = f"https://finviz.com/quote.ashx?t={tkr}"
        resp = requests.get(url, headers=hdr, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        return [a.get_text(strip=True) for a in soup.select("table.fullview-news-outer td a")][:n]
    except Exception:
        return []

def headlines(tkr, n=60, sources=None):
    sources = sources or ["Yahoo", "Google"]
    h = []
    if "Yahoo" in sources:       h.extend(news_yf(tkr, n))
    if "Google" in sources:      h.extend(news_google(tkr, n))
    if "Reuters" in sources:     h.extend(news_reuters(tkr, n))
    if "MarketWatch" in sources: h.extend(news_marketwatch(tkr, n))
    if "Investing" in sources:   h.extend(news_investing(tkr, n))
    if "Finviz" in sources:      h.extend(news_finviz(tkr, n))
    seen, uniq = set(), []
    for txt in h:
        if txt not in seen:
            uniq.append(txt)
            seen.add(txt)
        if len(uniq) >= n:
            break
    return uniq

# ───────── ANÁLISE DE SENTIMENTO COM FINBERT ─────────
def _classify(text):
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"

    if lang != "en" or len(text.split()) < 4:
        try:
            text = traduzir_pt_para_en(text)
        except Exception as e:
            print(f"⚠️ Tradução falhou: {e}")
            return 0, 0.0

    try:
        res = _pipe_finbert(text[:512])[0]
        label, conf = res["label"].lower(), res["score"]
        score = {"positive": 1, "neutral": 0, "negative": -1}.get(label, 0)
        return score, conf
    except Exception as e:
        print(f"⚠️ Erro na classificação FinBERT: {e}")
        return 0, 0.0


def sentimento_ticker(ticker, n_headlines=20, sources=None):
    heads = headlines(ticker, n_headlines, sources)
    if not heads:
        return {"ticker": ticker, "sentimento": "indefinido", "headlines": []}
    rows = []
    for h in heads:
        sc, cf = _classify(h)
        rows.append({"headline": h, "score": sc, "conf": cf})
    df = pd.DataFrame(rows)

    top_positivas = df[df["score"] > 0].sort_values("conf", ascending=False).head(2)
    top_negativas = df[df["score"] < 0].sort_values("conf", ascending=False).head(2)

    agg = (df["score"] * df["conf"]).sum() / df["conf"].sum()
    if agg > 0.2:
        senti = "positivo"
    elif agg < -0.2:
        senti = "negativo"
    else:
        senti = "neutro"
    return {
        "ticker": ticker,
        "sentimento": senti,
        "score_ponderado": round(agg, 3),
        "detalhes": df.sort_values("conf", ascending=False)
    }

# ───────── INTERFACE ─────────
def top_sentiment_interface(universe, top_k, n_head,
                            yahoo_selected, google_selected,
                            reuters_selected, marketwatch_selected,
                            investing_selected, finviz_selected):
    clear_output()
    selected_sources = []
    if yahoo_selected:       selected_sources.append("Yahoo")
    if google_selected:      selected_sources.append("Google")
    if reuters_selected:     selected_sources.append("Reuters")
    if marketwatch_selected: selected_sources.append("MarketWatch")
    if investing_selected:   selected_sources.append("Investing")
    if finviz_selected:      selected_sources.append("Finviz")

    if not selected_sources:
        print("⚠️ Nenhuma fonte selecionada.")
        return

    print(f"\n🌎 Universo: {universe.upper()} | 🏆 Top {top_k} | 📰 {n_head} manchetes por ativo")
    print(f"📡 Fontes: {', '.join(selected_sources)}")

    tickers = UNIVERSE_MAP[universe]()
    results = []

    for t in tqdm(tickers, total=len(tickers), desc=f"Analisando {universe.upper()}"):
        score = sentimento_ticker(t, n_head, selected_sources)

        resumo = sentimento_ticker(t, n_head, selected_sources)
        detalhes = resumo.get("detalhes", pd.DataFrame())

        if isinstance(score, dict):
            score = score.get("score_ponderado")
        if score is None:
            continue
        results.append({"ticker": t, "score": round(score, 3)})

    if not results:
        print("⚠️ Nenhum resultado válido.")
        return

    df = pd.DataFrame(results).sort_values("score", ascending=False).head(top_k)
    df.to_csv(f"sentimento_{universe}.csv", index=False)
    print(f"\n💾 Resultado exportado para: sentimento_{universe}.csv")
    print(f"\n🏆 TOP {top_k} ({universe.upper()})")
    print(df.to_string(index=False, formatters={"score": "{:+.3f}".format}))

# ───────── WIDGETS ─────────
universe_widget = widgets.Dropdown(options=["ibov", "sp500", "nasdaq"], value="ibov", description="🌎 Universo:")
top_k_widget = widgets.IntSlider(value=5, min=1, max=20, step=1, description="🏆 Top K:")
n_head_widget = widgets.IntSlider(value=100, min=10, max=200, step=10, description="📰 Manchetes:")

checkboxes = [
    widgets.Checkbox(value=True, description="Yahoo Finance"),
    widgets.Checkbox(value=True, description="Google News"),
    widgets.Checkbox(value=True, description="Reuters"),
    widgets.Checkbox(value=True, description="MarketWatch"),
    widgets.Checkbox(value=True, description="Investing.com"),
    widgets.Checkbox(value=True, description="Finviz"),
]

run_button = widgets.Button(description="🚀 Executar análise")

def on_button_clicked(b):
    top_sentiment_interface(
        universe_widget.value,
        top_k_widget.value,
        n_head_widget.value,
        *[cb.value for cb in checkboxes]
    )

run_button.on_click(on_button_clicked)

display(universe_widget, top_k_widget, n_head_widget,
        widgets.Label("📡 Fontes de Notícias:"),
        *checkboxes, run_button)
