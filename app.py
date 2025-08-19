import streamlit as st
import pandas as pd
import numpy as np
import requests, io, json
import re  # ДОБАВИТЬ
from datetime import datetime, timezone
from io import BytesIO

# PDF (можно оставить как есть; если когда-то будет проблема - сделаем try/except)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

st.set_page_config(page_title="Coffee Landed Cost — MVP", page_icon="☕", layout="wide")
st.title("☕ Coffee Landed Cost — MVP")

# ---------- Helpers (ВСЕ функции ОБЯЗАТЕЛЬНО до сайдбара) ----------

def arabica_centlb_to_usd_per_kg(cents_per_lb: float) -> float:
    return (cents_per_lb / 100.0) / 0.45359237

def robusta_usd_per_tonne_to_usd_per_kg(usd_per_tonne: float) -> float:
    return usd_per_tonne / 1000.0

@st.cache_data(ttl=600)
def fetch_stooq_csv(symbol: str, interval: str = "d") -> pd.DataFrame:
    """CSV с Stooq с простым user-agent и двумя доменами."""
    headers = {"User-Agent": "Mozilla/5.0"}
    for base in ["https://stooq.com", "https://stooq.pl"]:
        url = f"{base}/q/d/l/?s={symbol}&i={interval}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.ok and r.text.strip():
            first_line = r.text.splitlines()[0].lower()
            if "date" in first_line and "close" in first_line:
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip().lower() for c in df.columns]
                return df.dropna()
    raise RuntimeError("Stooq CSV not available")

def def fetch_yahoo_intraday_last(symbol: str, interval: str = "15m", range_: str = "1d"):
    """
    Возвращает (last_price, last_timestamp_utc) из интрадей-данных Yahoo.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": range_, "interval": interval}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]

    closes = j["indicators"]["quote"][0]["close"] or []
    ts = j.get("timestamp") or []
    last = None
    last_ts = None
    for i in range(len(closes) - 1, -1, -1):
        if closes[i] is not None:
            last = float(closes[i])
            last_ts = ts[i] if i < len(ts) else None
            break
    if last is None:
        last = float(j["meta"]["regularMarketPrice"])
        last_ts = j["meta"].get("currentTradingPeriod", {}).get("regular", {}).get("end", None)

    return last, last_ts
    
def stooq_front_series_symbol(base_symbol: str = "rm.f") -> str | None:
    """
    Открываем страницу серий Stooq и вытаскиваем код активной серии, напр. RMU25.F
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://stooq.com/q/f/?s={base_symbol}"
    r = requests.get(url, headers=headers, timeout=10)
    if not r.ok or not r.text:
        return None
    m = re.search(r"RM[A-Z]\d{2}\.F", r.text, re.IGNORECASE)
    return m.group(0).upper() if m else None

@st.cache_data(ttl=900)  # 15 минут
def get_live_prices(seed: int = 0):
    data = {}

    # ----- Arabica (¢/lb) -----
    try:
        kc = fetch_stooq_csv("kc.f")
        last_kc = float(kc.iloc[-1]["close"])
        data["KC.F"] = {
            "last_raw": last_kc,
            "unit": "¢/lb",
            "usdkg": arabica_centlb_to_usd_per_kg(last_kc),
            "source": "Stooq",
            "asof": None,
        }
    except Exception:
        try:
            last_kc, last_ts = fetch_yahoo_intraday_last("KC=F", interval="15m", range_="1d")
            data["KC.F"] = {
                "last_raw": last_kc,
                "unit": "¢/lb",
                "usdkg": arabica_centlb_to_usd_per_kg(last_kc),
                "source": "Yahoo 15m",
                "asof": last_ts,
            }
        except Exception as e:
            data["KC.F"] = {"error": f"Arabica: {e}"}

    # ----- Robusta (USD/tonne) -----
    try:
        # Stooq непрерывный символ
        rm = fetch_stooq_csv("rm.f")
        last_rm = float(rm.iloc[-1]["close"])
        data["RM.F"] = {
            "last_raw": last_rm,
            "unit": "USD/t",
            "usdkg": robusta_usd_per_tonne_to_usd_per_kg(last_rm),
            "source": "Stooq",
            "asof": None,
        }
    except Exception:
        # Stooq front-month серия (напр., RMU25.F)
        try:
            front = stooq_front_series_symbol("rm.f")
            if front:
                rm_series = fetch_stooq_csv(front)
                last_rm = float(rm_series.iloc[-1]["close"])
                data["RM.F"] = {
                    "last_raw": last_rm,
                    "unit": "USD/t",
                    "usdkg": robusta_usd_per_tonne_to_usd_per_kg(last_rm),
                    "source": f"Stooq ({front})",
                    "asof": None,
                }
            else:
                raise RuntimeError("No Stooq front month found")
        except Exception:
            # Yahoo intraday (тикер может быть RC=F или RM=F)
            last_rm = None
            last_ts = None
            yahoo_sym = None
            last_err = None
            for ysym in ("RC=F", "RM=F"):
                try:
                    yprice, yts = fetch_yahoo_intraday_last(ysym, interval="15m", range_="1d")
                    last_rm, last_ts, yahoo_sym = float(yprice), yts, ysym
                    break
                except Exception as e:
                    last_err = e
            if last_rm is not None:
                data["RM.F"] = {
                    "last_raw": last_rm,
                    "unit": "USD/t",
                    "usdkg": robusta_usd_per_tonne_to_usd_per_kg(last_rm),
                    "source": f"Yahoo 15m ({yahoo_sym})",
                    "asof": last_ts,
                }
            else:
                data["RM.F"] = {"error": f"Robusta: {last_err}"}

    data["ts"] = datetime.now(timezone.utc).isoformat()
    return data

# ---------- Sidebar: market (ОДИН раз) ----------
def fmt_ts(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M UTC") if ts else "n/a"

with st.sidebar:
    if "refresh_seed" not in st.session_state:
        st.session_state["refresh_seed"] = 0
    if st.button("↻ Обновить котировки"):
        st.session_state["refresh_seed"] += 1

    st.subheader("Рынок (бесплатные квоты) — Stooq/Yahoo")
    data = get_live_prices(st.session_state["refresh_seed"])
    kc = data.get("KC.F", {})
    rm = data.get("RM.F", {})

    colA, colB = st.columns(2)
    with colA:
        if "error" in kc:
            st.error(kc["error"])
        else:
            st.metric("Arabica (KC)", f"{kc['last_raw']:.2f} {kc.get('unit','')}")
            st.caption(f"≈ {kc['usdkg']:.3f} $/кг • {kc.get('source','?')} • as of {fmt_ts(kc.get('asof'))}")
    with colB:
        if "error" in rm:
            st.error(rm["error"])
        else:
            st.metric("Robusta (RM)", f"{rm['last_raw']:.2f} {rm.get('unit','')}")
            st.caption(f"≈ {rm['usdkg']:.3f} $/кг • {rm.get('source','?')} • as of {fmt_ts(rm.get('asof'))}")

    st.caption("Квоты с задержкой. Для сделок сверяйте с брокером/биржей.")
    
with st.sidebar:
    st.subheader("Рынок (бесплатные квоты) — Stooq/Yahoo")
    data = get_live_prices()
    kc = data.get("KC.F", {})
    rm = data.get("RM.F", {})
    colA, colB = st.columns(2)
    with colA:
        if "error" in kc:
            st.error(kc["error"])
        else:
            st.metric("Arabica (KC)", f"{kc['last_raw']:.2f} {kc.get('unit','')}")
            st.caption(f"≈ {kc['usdkg']:.3f} $/кг • Source: {kc.get('source','?')}")
    with colB:
        if "error" in rm:
            st.error(rm["error"])
        else:
            st.metric("Robusta (RM)", f"{rm['last_raw']:.2f} {rm.get('unit','')}")
            st.caption(f"≈ {rm['usdkg']:.3f} $/кг • Source: {rm.get('source','?')}")
    st.caption("Квоты с задержкой. Для сделок сверяйте с брокером/биржей.")

# ---------- Stage A & B Starters ----------
st.header("Калькулятор")

src = st.radio("Источник цены", ["Онлайн фьючерс (Stooq/Yahoo)", "Введу вручную"], horizontal=True)

col1, col2, col3 = st.columns(3)
with col1:
    instrument = st.selectbox("Инструмент", ["Arabica (KC.F)", "Robusta (RM.F)"])
with col2:
    # base price $/kg from market or manual
    if src == "Онлайн фьючерс (Stooq)":
        market_usdkg = (data.get("KC.F", {}).get("usdkg") if instrument.startswith("Arabica")
                        else data.get("RM.F", {}).get("usdkg"))
        if market_usdkg is None:
            st.warning("Нет данных рынка — переключитесь на ввод вручную.")
            base_usdkg = st.number_input("Базовая цена $/кг", min_value=0.0, value=3.000, step=0.001)
        else:
            st.text_input("Базовая цена $/кг (из рынка)", value=f"{market_usdkg:.4f}", disabled=True)
            base_usdkg = float(market_usdkg)
    else:
        base_usdkg = st.number_input("Базовая цена $/кг", min_value=0.0, value=3.000, step=0.001)
with col3:
    diff = st.number_input("Дифференциал $/кг (±)", value=0.000, step=0.010, help="Добавка/скидка к базовой цене")
effective_usdkg = base_usdkg + diff
st.caption(f"Эффективная цена: **{effective_usdkg:.4f} $/кг**")

# Containers
st.markdown("**Контейнер / вес**")
colw1, colw2, colw3 = st.columns(3)
with colw1:
    container = st.selectbox("Контейнер", ["20' (~19.2 т)", "40' (~26.0 т)", "Пользовательский"])
with colw2:
    if container.startswith("20"):
        weight_kg = st.number_input("Вес (кг)", min_value=1.0, value=19200.0, step=100.0)
    elif container.startswith("40"):
        weight_kg = st.number_input("Вес (кг)", min_value=1.0, value=26000.0, step=100.0)
    else:
        weight_kg = st.number_input("Вес (кг)", min_value=1.0, value=1000.0, step=100.0)
with colw3:
    incoterm = st.selectbox("Incoterm", ["FOB","CFR","CIF"], help="Упрощённая логика")

# Jurisdiction presets
st.markdown("**Юрисдикция (НДС/пошлина пресет)**")
jur_data = json.load(open("data/jurisdictions.json","r"))
colsj1, colsj2 = st.columns(2)
with colsj1:
    jname = st.selectbox("Страна/регион", list(jur_data.keys()))
with colsj2:
    j = jur_data[jname]
    vat_rate = st.number_input("Ставка НДС (0..1)", min_value=0.0, max_value=1.0, value=float(j["vat_rate"]), step=0.01)
duty_rate = st.number_input("Пошлина (адвал., 0..1)", min_value=0.0, max_value=1.0, value=float(j["duty_rate"]), step=0.01)
duty_sp_perkg = st.number_input("Пошлина (специф., $/кг)", min_value=0.0, value=0.0, step=0.01)

# Route presets (for CFR/CIF)
st.markdown("**Маршрут (для CFR/CIF пресеты фрахта/страховки)**")
routes = json.load(open("data/routes.json","r"))
route_names = ["(не использ.)"] + list(routes.keys())
colr1, colr2, colr3 = st.columns(3)
with colr1:
    rsel = st.selectbox("Маршрут", route_names)
with colr2:
    if rsel != "(не использ.)" and incoterm in routes[rsel]["incoterms"]:
        freight = st.number_input("Фрахт (USD)", min_value=0.0, value=float(routes[rsel]["freight"]), step=10.0)
    else:
        freight = st.number_input("Фрахт (USD)", min_value=0.0, value=1800.0, step=10.0)
with colr3:
    if rsel != "(не использ.)" and incoterm in routes[rsel]["incoterms"]:
        insurance = st.number_input("Страховка (USD)", min_value=0.0, value=float(routes[rsel]["insurance"]), step=5.0)
    else:
        insurance = st.number_input("Страховка (USD)", min_value=0.0, value=80.0, step=5.0)

# Fees editor (single line starter; can extend to multiple later)
st.markdown("**Локальные сборы (стартер)**")
feec1, feec2, feec3, feec4, feec5 = st.columns([2,1,1,1,1])
with feec1:
    fee_name = st.text_input("Название", value="Customs processing")
with feec2:
    fee_kind = st.selectbox("Тип", ["fixed","percent"])
with feec3:
    if fee_kind=="fixed":
        fee_amount = st.number_input("Сумма ($)", min_value=0.0, value=25.0, step=1.0)
        fee_rate = 0.0; fee_base="CV"
    else:
        fee_rate = st.number_input("Ставка (%)", min_value=0.0, value=0.0, step=0.1)/100.0
        fee_base = st.selectbox("База", ["CV","Goods","CVPlusDuty"])
        fee_amount = 0.0
with feec4:
    fee_vb = st.checkbox("Включать в базу НДС", value=True)
with feec5:
    st.write("")
fees = [ {"name": fee_name, "kind": fee_kind, "amount": fee_amount, "rate": fee_rate, "base": fee_base, "vat_base": fee_vb} ]

st.divider()

if st.button("Рассчитать", type="primary"):
    b = compute_quote(
        usd_per_kg=float(effective_usdkg),
        weight_kg=float(weight_kg),
        incoterm=incoterm,
        freight=float(freight),
        insurance=float(insurance),
        duty_rate=float(duty_rate),
        duty_sp_perkg=float(duty_sp_perkg),
        vat_rate=float(vat_rate),
        fees=fees
    )
    colL, colR = st.columns([1,1])
    with colL:
        st.subheader("Итог")
        st.metric("Landed total (USD)", f"{b['total']:.2f}")
        st.metric("$/кг", f"{b['per_kg']:.4f}")
        st.write(f"Incoterm: {incoterm} • Вес: {weight_kg:,.0f} кг • Регион: {jname}")
        st.write(f"Фрахт: {freight:.2f} • Страховка: {insurance:.2f} • Дифференциал: {diff:+.3f} $/кг")
    with colR:
        st.subheader("Разбор")
        st.dataframe(make_result_df(b), hide_index=True, use_container_width=True)
        st.write("Сборы:")
        st.dataframe(pd.DataFrame([{"Сбор": f["name"], "Сумма": f["amount"], "В базу НДС": f["vat_base"]} for f in b["fees"]]), hide_index=True, use_container_width=True)

    # Export buttons
    calc_params = {
        "Instrument": instrument,
        "Base USD/kg": f"{base_usdkg:.4f}",
        "Differential USD/kg": f"{diff:+.4f}",
        "Effective USD/kg": f"{effective_usdkg:.4f}",
        "Weight kg": weight_kg,
        "Incoterm": incoterm,
        "Freight USD": freight,
        "Insurance USD": insurance,
        "VAT rate": vat_rate,
        "Duty ad val.": duty_rate,
        "Duty specific $/kg": duty_sp_perkg,
        "Jurisdiction": jname,
        "Route": rsel
    }
    # CSV/Excel
    excel_buf = export_excel(b, calc_params)
    st.download_button("⬇️ Скачать Excel", data=excel_buf, file_name="landed_cost.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    # CSV
    df_main = make_result_df(b)
    csv = df_main.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Скачать CSV (результат)", data=csv, file_name="result.csv", mime="text/csv")
    # PDF
    pdf_buf = export_pdf(b, calc_params)
    st.download_button("⬇️ Скачать PDF (сводка)", data=pdf_buf, file_name="summary.pdf", mime="application/pdf")

    # "Copy" text block (manual copy)
    st.subheader("Скопировать расчёт (текст)")
    txt = f"""Landed cost: {b['total']:.2f} USD ({b['per_kg']:.4f} USD/kg)
Instrument: {instrument}, Base: {base_usdkg:.4f}, Diff: {diff:+.4f}, Effective: {effective_usdkg:.4f}
Weight: {weight_kg:,.0f} kg, Incoterm: {incoterm}, Freight: {freight:.2f}, Insurance: {insurance:.2f}
VAT rate: {vat_rate}, Duty ad val.: {duty_rate}, Duty specific: {duty_sp_perkg}
Jurisdiction: {jname}, Route: {rsel}
CV: {b['customs_value']:.2f}, Duty total: {b['duty_total']:.2f}, Fees: {b['fees_total']:.2f}, VAT: {b['vat']:.2f}
"""
    st.text_area("Текст:", value=txt, height=160)
    st.info("Выделите текст и нажмите ⌘+C / Ctrl+C для копирования.")
else:
    st.info("Заполните поля и нажмите «Рассчитать».")
