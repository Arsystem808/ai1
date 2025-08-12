import os, pathlib, sys
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE))

from core.data_loader import DataLoader
from core.strategy import compute_signal
from core.llm import build_rationale

st.set_page_config(page_title="Signal Desk — Pro", layout="wide")
st.title("Signal Desk — Pro")
st.caption("Источник котировок: Polygon → Yahoo → CSV. Логика закрыта. UI в стиле Trading Central.")

def _fmt(x: float) -> str:
    s = f"{float(x):.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s

def _orients(sig: dict):
    px = float(sig["entry"])
    ups = [sig.get("R1"), sig.get("R2"), sig.get("R3"), sig.get("upper_zone")]
    dns = [sig.get("S1"), sig.get("S2"), sig.get("S3"), sig.get("lower_zone")]
    ups = sorted([float(v) for v in ups if isinstance(v,(int,float)) and v>px])[:2]
    dns = sorted([float(v) for v in dns if isinstance(v,(int,float)) and v<px], reverse=True)[:2]
    return ups, dns

def _chart(df: pd.DataFrame, sig: dict, show_orients=True, height=420):
    fig = go.Figure([go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])])
    colors = {"Entry":"#2563eb","TP1":"#16a34a","TP2":"#16a34a","SL":"#dc2626",
              "Ключевая":"#6b7280","Верхняя зона":"#f59e0b","Нижняя зона":"#10b981",
              "Ориентир ↑":"#a78bfa","Ориентир ↓":"#f472b6"}
    lines = [("Entry",sig["entry"]),("TP1",sig["tp1"]),("TP2",sig["tp2"]),("SL",sig["sl"]),
             ("Ключевая",sig["key_mark"]),("Верхняя зона",sig["upper_zone"]),("Нижняя зона",sig["lower_zone"])]
    for name, y in lines:
        fig.add_hline(y=y, line_width=1, line_dash="dot", line_color=colors.get(name,"#999"),
                      annotation_text=name, annotation_position="top left")
    if show_orients:
        ups, dns = _orients(sig)
        if len(ups)>=1: fig.add_hline(y=ups[0], line_width=1, line_dash="dot", line_color=colors["Ориентир ↑"],
                                      annotation_text="Ориентир ↑1", annotation_position="top left")
        if len(ups)>=2: fig.add_hline(y=ups[1], line_width=1, line_dash="dot", line_color=colors["Ориентир ↑"],
                                      annotation_text="Ориентир ↑2", annotation_position="top left")
        if len(dns)>=1: fig.add_hline(y=dns[0], line_width=1, line_dash="dot", line_color=colors["Ориентир ↓"],
                                      annotation_text="Ориентир ↓1", annotation_position="top left")
        if len(dns)>=2: fig.add_hline(y=dns[1], line_width=1, line_dash="dot", line_color=colors["Ориентир ↓"],
                                      annotation_text="Ориентир ↓2", annotation_position="top left")
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=height, showlegend=False)
    return fig

# ── Верхняя панель ввода
default_tickers = os.getenv("DEFAULT_TICKERS", "QQQ,AAPL,MSFT,NVDA")
tickers = st.text_input("Тикеры (через запятые)", value=default_tickers).upper()
symbols = [t.strip() for t in tickers.split(",") if t.strip()]
try: idx = symbols.index("QQQ")
except ValueError: idx = 0 if symbols else 0
symbol = st.selectbox("Тикер", symbols if symbols else ["QQQ"], index=idx)

loader = DataLoader()

# ── Три вкладки горизонта как в TC
tabS, tabM, tabL = st.tabs(["Краткосрок", "Среднесрок", "Долгосрок"])
tabs = [("short", tabS), ("swing", tabM), ("position", tabL)]

# Загружаем данные один раз
try:
    fetched = loader.history(symbol, period="12mo", interval="1d")
    df = fetched.df
    source = fetched.source
except Exception as e:
    st.error(str(e))
    st.stop()

def render_block(horizon: str, container):
    with container:
        sig = compute_signal(df, symbol, horizon, params=None)
        sig["source"] = source

        # Карточка сигнала
        color = "#16a34a" if sig["action"]=="BUY" else ("#dc2626" if sig["action"]=="SHORT" else "#6b7280")
        st.markdown(
            f"<div style='padding:10px 12px;border:1px solid #e5e7eb;border-radius:10px'>"
            f"<div style='display:flex;gap:10px;align-items:center'>"
            f"<div style='background:{color};color:#fff;padding:4px 10px;border-radius:8px;font-weight:700'>{sig['action']}</div>"
            f"<div style='opacity:.7'>Источник: {sig['source']}</div>"
            f"<div style='margin-left:auto;opacity:.7'>Уверенность: {sig['confidence']:.2f}</div>"
            f"</div>"
            f"<div style='margin-top:8px;font-size:14.5px;color:#111'>"
            f"План: вход {_fmt(sig['entry'])}, стоп {_fmt(sig['sl'])}, цели {_fmt(sig['tp1'])} / {_fmt(sig['tp2'])}. "
            f"Рабочий коридор {_fmt(sig['lower_zone'])}–{_fmt(sig['upper_zone'])}, ключевая {_fmt(sig['key_mark'])}."
            f"</div></div>", unsafe_allow_html=True
        )

        # Метрики
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Entry", _fmt(sig["entry"])); m2.metric("TP1", _fmt(sig["tp1"]))
        m3.metric("TP2", _fmt(sig["tp2"]));     m4.metric("SL",  _fmt(sig["sl"]))

        # График
        st.plotly_chart(_chart(df, sig, show_orients=True, height=360), use_container_width=True)

        # Текст «как у TC»: коротко/стандарт/подробно
        style = st.radio("Формат комментария", ["Коротко","Стандарт","Подробно"], index=0, horizontal=True)
        st.write(build_rationale(symbol, {"short":"Краткосрок","swing":"Среднесрок","position":"Долгосрок"}[horizon], sig, style))

for hz, tb in tabs:
    render_block(hz, tb)
