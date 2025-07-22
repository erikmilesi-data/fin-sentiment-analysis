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
from tqdm.notebook import tqdm  # Barra de progresso amigável para Jupyter
import ipywidgets as widgets
from IPython.display import display, clear_output

# ───────── UNIVERSOS ─────────
def universe_ibov():
    hdr = {"User-Agent": "Mozilla/5.0"}
    url = ("https://sistemaswebb3-listados.b3.com.br/"
           "listedCompaniesPage/getIbovespaConstituents?language=pt-br")
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
    return [tr.find_all("td")[0].text.strip().replace(".", "-")
            for tr in tbl.find_all("tr")[1:]]

def universe_nasdaq():
    url = ("https://old.nasdaq.com/screening/"
           "companies-by-industry.aspx?exchange=NASDAQ&render=download")
    rdr = csv.DictReader(io.StringIO(requests.get(url, timeout=30).text))
    return [row["Symbol"] for row in rdr]

UNIVERSE_MAP = {"ibov": universe_ibov, "sp500": universe_sp500, "nasdaq": universe_nasdaq}

# ───────── NOTÍCIAS ─────────
def news_yf(tkr, n=60):
    try:
        ticker_obj = yf.Ticker(tkr)
        items = ticker_obj.news or []
    except Exception as e:
        if "404" in str(e):
            print(f"⚠️ Yahoo Finance: ticker {tkr} não encontrado. Pulando...")
        else:
            print(f"⚠️ Erro no Yahoo Finance para {tkr}: {e}")
        return []  # Retorna lista vazia para continuar
    hd = []
    for it in items:
        tx = it.get("title") or it.get("headline") or it.get("summary")
        if tx:
            hd.append(" ".join(tx.split()))
        if len(hd) >= n:
            break
    return hd


def company_name(tkr):
    try:
        info = yf.Ticker(tkr).info
        return info.get("shortName") or info.get("longName") or tkr
    except Exception:
        return tkr

def news_google(tkr, n=60):
    name = company_name(tkr)
    q = requests.utils.quote(f'"{name}" OR "{tkr}"')
    url = (f"https://news.google.com/rss/search?q={q}"
           f"&hl=pt-BR&gl=BR&ceid=BR:pt-419")
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
        headlines = [a.get_text(strip=True) for a in soup.select(".js-inner-all-results-quote-item .title")]
        return headlines[:n]
    except Exception as e:
        print(f"⚠️ Erro ao buscar Investing.com para {tkr}: {e}")
        return []

def news_finviz(tkr, n=60):
    try:
        hdr = {"User-Agent": "Mozilla/5.0"}
        url = f"https://finviz.com/quote.ashx?t={tkr}"
        resp = requests.get(url, headers=hdr, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = [a.get_text(strip=True) for a in soup.select("table.fullview-news-outer td a")]
        return headlines[:n]
    except Exception as e:
        print(f"⚠️ Erro ao buscar Finviz para {tkr}: {e}")
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
    # Remove duplicados
    seen, uniq = set(), []
    for txt in h:
        if txt not in seen:
            uniq.append(txt)
            seen.add(txt)
        if len(uniq) >= n:
            break
    return uniq

# Placeholder para classificadores
def _pipe_finbert(text): return [{"label": "positive", "score": 0.9}]
def _pipe_multi(text): return [{"label": "4 stars", "score": 0.7}]

def _classify(text):
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    if lang == "en":
        res = _pipe_finbert(text[:512])[0]
        lbl, conf = res["label"], res["score"]
        score = {"positive": 1, "neutral": 0, "negative": -1}.get(lbl, 0)
    else:
        res = _pipe_multi(text[:512])[0]
        lbl, conf = res["label"], res["score"]
        stars = int(lbl[0])
        score = (stars - 3) / 2
    return score, conf

def sentimento_ticker(ticker: str, n_headlines: int = 20, sources=None):
    heads = headlines(ticker, n_headlines, sources)
    if not heads:
        return {"ticker": ticker, "sentimento": "indefinido", "headlines": []}
    rows = []
    for h in heads:
        sc, cf = _classify(h)
        rows.append({"headline": h, "score": sc, "conf": cf})
    df = pd.DataFrame(rows)
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

# ───────── INTERFACE COM CHECKBOXES ─────────
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
        print("⚠️ Nenhuma fonte selecionada. Marque pelo menos uma.")
        return

    print(f"\n🌎 Universo selecionado: {universe.upper()}")
    print(f"🏆 Top {top_k} | 📰 {n_head} notícias por ativo")
    print(f"📡 Fontes escolhidas: {', '.join(selected_sources)}")

    tickers = UNIVERSE_MAP[universe]()
    results = []

    for t in tqdm(tickers, total=len(tickers), desc=f"Analisando {universe.upper()}"):
        score = sentimento_ticker(t, n_head, selected_sources)
        if isinstance(score, dict):
            score = score.get("score_ponderado")
        if score is None:
            continue
        results.append({"ticker": t, "score": round(score, 3)})

    if not results:
        print("⚠️ Nenhum resultado válido.")
        return

    df = (pd.DataFrame(results)
          .sort_values("score", ascending=False)
          .head(top_k))
    print(f"\n── 🏆 TOP {top_k} SCORES ({universe.upper()}) ──")
    print(df.to_string(index=False,
                       formatters={"score": "{:+.3f}".format}))

universe_widget = widgets.Dropdown(
    options=["ibov", "sp500", "nasdaq"],
    value="ibov",
    description="🌎 Universo:"
)
top_k_widget = widgets.IntSlider(value=5, min=1, max=20, step=1, description="🏆 Top K:")
n_head_widget = widgets.IntSlider(value=100, min=10, max=200, step=10, description="📰 Headlines:")

yahoo_checkbox       = widgets.Checkbox(value=True, description="Yahoo Finance")
google_checkbox      = widgets.Checkbox(value=True, description="Google News")
reuters_checkbox     = widgets.Checkbox(value=True, description="Reuters")
marketwatch_checkbox = widgets.Checkbox(value=True, description="MarketWatch")
investing_checkbox   = widgets.Checkbox(value=True, description="Investing.com")
finviz_checkbox      = widgets.Checkbox(value=True, description="Finviz")

run_button = widgets.Button(description="🚀 Executar análise")

def on_button_clicked(b):
    top_sentiment_interface(
        universe_widget.value,
        top_k_widget.value,
        n_head_widget.value,
        yahoo_checkbox.value,
        google_checkbox.value,
        reuters_checkbox.value,
        marketwatch_checkbox.value,
        investing_checkbox.value,
        finviz_checkbox.value
    )

run_button.on_click(on_button_clicked)

display(universe_widget, top_k_widget, n_head_widget,
        widgets.Label("📡 Escolha as fontes:"),
        yahoo_checkbox, google_checkbox, reuters_checkbox,
        marketwatch_checkbox, investing_checkbox, finviz_checkbox,
        run_button)
