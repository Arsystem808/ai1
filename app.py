# app.py
import os, pathlib, sys
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE))

from core.data_loader import DataLoader
from core.strategy import compute_signal
from core.llm import build_rationale

def _fmt_val(x: float) -> str:
    s = f"{float(x):.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s

def _fmt_range(lo: float, hi: float) -> str:
    return f"{_fmt_val(lo)}–{_fmt_val(hi)}"

def _neutral_orients(sig: dict) -> tuple[list[float], list[float]]:
    entry_px = float(sig["entry"])
    up_candidates = [sig.get("upper_zone"), sig.get("key_mark"), sig.get("R1"), sig.get("R2"), sig.get("R3")]
    down_candidates = [sig.get("lower_zone"), sig.get("key_mark"), sig.get("S1"), sig.get("S2"), sig.get("S3")]
    ups = sorted([float(x) for x in up_candidates if isinstance(x,(int,float)) and x is not None and x > entry_px])[:2]
    dns = sorted([float(x) for x in down_candidates if isinstance(x,(int,float)) and x is not None and x < entry_px], reverse=True)[:2]
    return ups, dns

def _infer_zones_for_text(sig: dict) -> tuple[str, str]:
    ups, dns = _neutral_orients(sig)
    entry = float(sig["entry"])
    if len(dns) >= 2:
        wait_lo, wait_hi = min(dns[0], dns[1]), max(dns[0], dns[1])
    elif len(dns) == 1:
        alt = float(sig.get("lower_zone", dns[0]))
        wait_lo, wait_hi = min(dns[0], alt), max(dns[0], alt)
    else:
        lz = float(sig.get("lower_zone", entry*0.98))
        km = float(sig.get("key_mark", entry))
        wait_lo, wait_hi = min(lz, km), max(lz, km)

    if len(ups) >= 2:
        short_lo, short_hi = min(ups[0], ups[1]), max(ups[0], ups[1])
    elif len(ups) == 1:
        tp1 = float(sig.get("tp1", entry*1.01))
        step = max(abs(tp1 - entry) * 0.3, 0.5)
        short_lo, short_hi = ups[0], ups[0] + step
        if short_lo > short_hi: short_lo, short_hi = short_hi, short_lo
    else:
        uz = float(sig.get("upper_zone", entry*1.02))
        km = float(sig.get("key_mark", entry))
        short_lo, short_hi = min(km, uz), max(km, uz)
    return _fmt_range(wait_lo, wait_hi), _fmt_range(short_lo, short_hi)

st.set_page_config(page_title="AI Trading — Clean", layout="wide")
st.title("AI Trading — Clean")
st.caption("Данные: Polygon → Yahoo → CSV. Без калибровки, без скользящих средних. Текст — офлайн, без раскрытия методики.")

# Тикеры и горизонт
default_tickers = os.getenv("DEFAULT_TICKERS", "QQQ,AAPL,MSFT,NVDA")
tickers = st.text_input("Tickers (через запятые)", value=default_tickers).upper()
symbols = [t.strip() for t in tickers.split(",") if t.strip()]

h_map = {"Краткосрок":"short","Среднесрок":"swing","Долгосрок":"position"}
horizon_ui = st.selectbox("Горизонт", list(h_map.keys()), index=1)
horizon = h_map[horizon_ui]

detail = st.selectbox("Степень детализации описания", ["Коротко", "Стандарт", "Подробно"], index=0)

try:
    idx = symbols.index("QQQ")
except ValueError:
    idx = 0 if symbols else 0
symbol = st.selectbox("Тикер", symbols if symbols else ["QQQ"], index=idx)

loader = DataLoader()

colA, colB = st.columns([1,2])
with colA:
    if st.button("Сгенерировать сигнал"):
        try:
            fetched = loader.history(symbol, period="6mo", interval="1d")
            st.session_state["source"] = fetched.source
            st.session_state["df"] = fetched.df
            sig = compute_signal(fetched.df, symbol, horizon, params=None)  # фикс. параметры
            sig["source"] = fetched.source
            wait_zone, short_zone = _infer_zones_for_text(sig)
            sig["wait_zone"] = wait_zone; sig["short_zone"] = short_zone
            st.session_state["signal"] = sig
        except Exception as e:
            st.error(str(e))

with colB:
    sig = st.session_state.get("signal")
    df: pd.DataFrame | None = st.session_state.get("df")
    source = st.session_state.get("source", "—")

    if sig and df is not None:
        st.subheader(f"{sig['symbol']} — {horizon_ui} | источник: {source}")
        color = "#16a34a" if sig["action"]=="BUY" else ("#dc2626" if sig["action"]=="SHORT" else "#6b7280")
        st.markdown(f"<div style='display:inline-block;padding:6px 10px;border-radius:8px;background:{color};color:white;font-weight:600'>{sig['action']}</div>", unsafe_allow_html=True)

        st.write("")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Entry", _fmt_val(sig["entry"]))
        m2.metric("TP1", _fmt_val(sig["tp1"]))
        m3.metric("TP2", _fmt_val(sig["tp2"]))
        m4.metric("SL", _fmt_val(sig["sl"]))

        d1, d2, d3 = st.columns(3)
        d1.metric("Ключевая отметка", _fmt_val(sig["key_mark"]))
        d2.metric("Верхняя зона", _fmt_val(sig["upper_zone"]))
        d3.metric("Нижняя зона", _fmt_val(sig["lower_zone"]))
        st.metric("Confidence", f"{sig['confidence']:.2f}")

        # График + нейтральные ориентиры ↑/↓
        ups, dns = _neutral_orients(sig)

        fig = go.Figure([go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]
        )])

        plan_lines = {"Entry": sig["entry"], "TP1": sig["tp1"], "TP2": sig["tp2"], "SL": sig["sl"],
                      "Ключевая отметка": sig["key_mark"], "Верхняя зона": sig["upper_zone"], "Нижняя зона": sig["lower_zone"]}
        colors = {"Entry":"#2563eb","TP1":"#16a34a","TP2":"#16a34a","SL":"#dc2626",
                  "Ключевая отметка":"#6b7280","Верхняя зона":"#f59e0b","Нижняя зона":"#10b981",
                  "Ориентир ↑1":"#a78bfa","Ориентир ↑2":"#a78bfa","Ориентир ↓1":"#f472b6","Ориентир ↓2":"#f472b6"}

        for label, y in plan_lines.items():
            if y is None: continue
            fig.add_hline(y=y, line_width=1, line_dash="dot", line_color=colors.get(label, "#999"),
                          annotation_text=label, annotation_position="top left")

        if len(ups) >= 1:
            fig.add_hline(y=ups[0], line_width=1, line_dash="dot", line_color=colors["Ориентир ↑1"],
                          annotation_text="Ориентир ↑1", annotation_position="top left")
        if len(ups) >= 2:
            fig.add_hline(y=ups[1], line_width=1, line_dash="dot", line_color=colors["Ориентир ↑2"],
                          annotation_text="Ориентир ↑2", annotation_position="top left")
        if len(dns) >= 1:
            fig.add_hline(y=dns[0], line_width=1, line_dash="dot", line_color=colors["Ориентир ↓1"],
                          annotation_text="Ориентир ↓1", annotation_position="top left")
        if len(dns) >= 2:
            fig.add_hline(y=dns[1], line_width=1, line_dash="dot", line_color=colors["Ориентир ↓2"],
                          annotation_text="Ориентир ↓2", annotation_position="top left")

        try:
            x0 = df["Date"].iloc[-min(len(df), 40)]; x1 = df["Date"].iloc[-1]
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=min(sig["entry"], sig["tp2"]), y1=max(sig["entry"], sig["tp2"]),
                          fillcolor="rgba(34,197,94,0.08)", line=dict(width=0), layer="below")
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=min(sig["sl"], sig["entry"]), y1=max(sig["sl"], sig["entry"]),
                          fillcolor="rgba(239,68,68,0.08)", line=dict(width=0), layer="below")
        except Exception:
            pass

        fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=460, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Текст
        if detail == "Коротко":
            text = (f"**🧠 {sig['symbol']} — {horizon_ui}**\n"
                    f"Цена упёрлась в зону, где ранее появлялась слабость; импульс выдыхается.\n\n"
                    f"**✅ Рекомендация:** {sig['action']}\n"
                    f"- Ждём отката к {sig['wait_zone']} для поиска лонга.\n"
                    f"- Агрессивный шорт допустим от {sig['short_zone']}, цели: {_fmt_val(sig['tp1'])}/{_fmt_val(sig['tp2'])}, стоп — > {_fmt_val(sig['sl'])}.\n\n"
                    f"💬 Опытным: можно работать от отката; остальным — пауза.")
            st.markdown(text)
        else:
            text = build_rationale(sig["symbol"], horizon_ui, sig, detail=detail)
            st.write(text)
    else:
        st.info("Нажмите «Сгенерировать сигнал». Если Polygon/Yahoo недоступны — возьмём demo CSV.")

