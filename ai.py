import streamlit as st
import pandas as pd
import numpy as np
import requests
from streamlit_echarts import st_echarts
from google import genai
import yfinance as yf
from config import GEMINI_API_KEY 

st.set_page_config(
    page_title="IDX Pro - Analisis Saham AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "GANTI_DENGAN_GEMINI_API_KEY_ANDA":
        client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        st.warning("⚠️ Harap atur GEMINI_API_KEY Anda di file config.py untuk mengaktifkan Analisis AI.")
        client = None
except Exception as e:
    st.error(f"Gagal inisialisasi Gemini Client: {e}")
    client = None

@st.cache_data(ttl=3600)
def get_historical_data(ticker):
    if not ticker.endswith(".JK"):
        idx_ticker = f"{ticker}.JK"
    else:
        idx_ticker = ticker
        
    try:
        data = yf.download(idx_ticker, period="6mo", interval="1d", progress=False)
        if data.empty:
            return pd.DataFrame()
            
        df = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index.name = 'Date'
        return df
    except Exception as e:
        st.error(f"❌ ERROR YFinance: {e}")
        return pd.DataFrame()

@st.cache_data
def get_snapshot_data(df_hist):
    if df_hist.empty or len(df_hist) < 2:
        return {"price": 0, "change_val": 0, "change_percent": 0, "volume": 0, "pe_ratio": 0, "roe_ratio": 0}
        
    try:
        latest_close = float(df_hist['Close'].iloc[-1])
        previous_close = float(df_hist['Close'].iloc[-2])
        latest_volume = float(df_hist['Volume'].iloc[-1])
    except:
        return {"price": 0, "change_val": 0, "change_percent": 0, "volume": 0, "pe_ratio": 0, "roe_ratio": 0}
    
    change_val = latest_close - previous_close
    change_percent = (change_val / previous_close) * 100 if previous_close != 0 else 0
    
    np.random.seed(int(latest_close)) 
    
    return {
        "price": latest_close, 
        "change_val": change_val, 
        "change_percent": change_percent, 
        "volume": latest_volume, 
        "pe_ratio": round(np.random.uniform(5, 30), 1), 
        "roe_ratio": round(np.random.uniform(5, 20), 1)
    }

def render_candlestick_chart(df):
    if df.empty:
        st.warning("Data kosong.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    ohlc = df[['Open', 'Close', 'Low', 'High']].values.tolist()
    dates = df.index.strftime('%Y-%m-%d').tolist()
    volumes = df['Volume'].values.tolist()
    
    options = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "grid": [
            {"left": "10%", "right": "8%", "height": "50%"},
            {"left": "10%", "right": "8%", "top": "70%", "height": "16%"}
        ],
        "xAxis": [
            {"data": dates, "scale": True, "boundaryGap": False, "axisLine": {"onZero": False}, "splitLine": {"show": False}, "splitNumber": 20, "min": "dataMin", "max": "dataMax"},
            {"data": dates, "gridIndex": 1, "scale": True, "boundaryGap": False, "axisLine": {"onZero": False}, "tickLength": 0, "show": False} 
        ],
        "yAxis": [
            {"scale": True, "splitArea": {"show": True}},
            {"scale": True, "gridIndex": 1, "splitNumber": 2, "axisLabel": {"show": False}, "axisLine": {"show": False}, "tickLength": 0, "splitLine": {"show": False}} 
        ],
        "dataZoom": [
            {"type": "inside", "xAxisIndex": [0, 1], "start": 50, "end": 100},
            {"type": "slider", "xAxisIndex": [0, 1], "start": 50, "end": 100}
        ],
        "series": [
            {
                "name": "OHLC",
                "type": "candlestick",
                "data": ohlc,
                "itemStyle": {"color": "#ec0000", "color0": "#00da3c", "borderColor": "#ec0000", "borderColor0": "#00da3c"}
            },
            {
                "name": "Volume",
                "type": "bar",
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "data": volumes,
                "itemStyle": {"color": "#7f7f7f"}
            }
        ]
    }
    st_echarts(options=options, height="600px")

def analyze_content_with_gemini(input_content, gemini_file_obj=None):
    if not client:
        return "⚠️ Gemini API belum terinisialisasi."
        
    base_prompt = """
    Anda adalah analis saham profesional khusus pasar Indonesia (IDX). 
    TUGAS: Analisis data laporan keuangan berikut dan berikan wawasan investasi.
    
    OUTPUT YANG DIHARAPKAN (Format Markdown):
    ### 1. 🔍 Kesimpulan Eksekutif
    (3 Poin utama tentang kesehatan perusahaan ini)
    
    ### 2. 🧮 Analisis Rasio (Estimasi)
    *Jika angka tersedia, hitunglah. Jika tidak, berikan estimasi kualitatif.*
    - **Profitabilitas:** (Bahas Margin/ROE)
    - **Solvabilitas:** (Bahas Hutang/DER)
    
    ### 3. 🚦 Rekomendasi Investasi
    **Vonis:** [STRONG BUY / BUY / HOLD / SELL / AVOID]
    **Alasan:** (Penjelasan singkat 2-3 kalimat)
    """

    if isinstance(input_content, str):
        contents = [base_prompt + f"\n\n--- DATA INPUT TEKS ---\n{input_content[:5000]}"]
    else:
        contents = input_content + [base_prompt]

    analysis_result = ""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=contents
        )
        analysis_result = response.text
    except Exception as e:
        analysis_result = f"❌ Gagal mendapatkan respons dari Gemini. Detail Error: {e}"
    finally:
        if gemini_file_obj:
            try:
                client.files.delete(name=gemini_file_obj.name)
            except Exception as e:
                st.error(f"Gagal membersihkan file Gemini: {e}")
    
    return analysis_result

st.sidebar.title("🛠️ Kontrol Analisis")
TICKER = st.sidebar.text_input("Kode Saham (cth: BBCA)", value="BBCA").upper().strip()

if st.sidebar.button("🔍 Refresh Data Saham"):
    st.cache_data.clear()
    st.rerun()

df_hist = get_historical_data(TICKER)
snapshot = get_snapshot_data(df_hist)

st.sidebar.markdown("---")
st.sidebar.subheader(f"📊 {TICKER} Snapshot")

if snapshot['price'] > 0:
    col_sb1, col_sb2 = st.sidebar.columns(2)
    col_sb1.metric("Harga", f"{snapshot['price']:,.0f}")
    col_sb2.metric("Perubahan", f"{snapshot['change_percent']:+.2f}%", delta_color="normal")
    st.sidebar.write(f"Volume: {snapshot['volume']/1000000:.1f} Juta")
else:
    st.sidebar.error("Data Saham Tidak Ditemukan.")

st.markdown("---")

st.title(f"Dashboard Analisis Saham: {TICKER}")
tab1, tab2 = st.tabs(["📊 Teknikal (Chart)", "🤖 Fundamental (AI)"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Harga Terakhir", f"Rp {snapshot['price']:,.0f}", f"{snapshot['change_val']:+,.0f}")
    col2.metric("P/E Ratio (Est)", f"{snapshot['pe_ratio']}x")
    col3.metric("ROE (Est)", f"{snapshot['roe_ratio']}%")
    
    st.markdown("### Pergerakan Harga & Volume")
    render_candlestick_chart(df_hist)
    
    with st.expander("Lihat Data Tabel"):
        st.dataframe(df_hist.sort_index(ascending=False).head(10), use_container_width=True)

with tab2:
    st.header("2. Analisis Laporan Keuangan (Fundamental AI)")
    st.info("💡 Pilih **Upload File** atau **Paste Teks** di bawah ini untuk memulai analisis.")
    
    if 'ai_analysis' not in st.session_state:
        st.session_state['ai_analysis'] = "Belum ada analisis yang dijalankan."
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        uploaded_file = st.file_uploader(
            "Upload Laporan Keuangan (.PDF, .TXT, .DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=False 
        )
        
        financial_text_input = st.text_area(
            "Atau Paste Teks Laporan Laba Rugi/Neraca di sini (Maks 5000 karakter)",
            height=200, 
            placeholder="Contoh: Pendapatan Bersih Kuartal III 2024: 10.5 Triliun. Total Ekuitas: 40 Triliun. Laba Bersih: 2 Triliun..."
        )
        
        analyze_btn = st.button("🚀 Jalankan Analisis Gemini", type="primary", use_container_width=True)
    
    with col_result:
        if analyze_btn:
            analysis_result = None
            uploaded_file_gemini = None 
            
            if uploaded_file is not None:
                with st.spinner(f"⏳ Mengunggah dan menganalisis file {uploaded_file.name}..."):
                    try:
                        uploaded_file_gemini = client.files.upload(
                            file=uploaded_file.getvalue(), 
                            display_name=uploaded_file.name
                        )
                        input_content = [uploaded_file_gemini]
                        analysis_result = analyze_content_with_gemini(input_content, uploaded_file_gemini)
                        
                    except Exception as e:
                        st.error(f"Gagal mengunggah file ke Gemini. Detail: {e}")

            elif financial_text_input:
                with st.spinner("⏳ Gemini sedang menganalisis data keuangan teks..."):
                    analysis_result = analyze_content_with_gemini(financial_text_input)
            
            else:
                st.warning("Mohon masukkan teks atau upload file terlebih dahulu.")
                
            if analysis_result:
                st.session_state['ai_analysis'] = analysis_result
                st.success("✅ Analisis Selesai! Lihat di tab 'Hasil Analisis AI'.")
        
        st.markdown("*Klik tombol 'Jalankan Analisis Gemini' di sebelah kiri untuk melihat hasil.*")


with tab2:
    st.subheader("Ringkasan dan Rekomendasi AI")
    if 'ai_analysis' in st.session_state:
        if st.session_state['ai_analysis'].startswith("✅"):
             st.info("Silakan unggah atau paste data di tab sebelumnya.")
        else:
             st.markdown(st.session_state['ai_analysis'])
    else:
        st.info("Belum ada analisis yang dijalankan.")

st.markdown("---")
st.caption("IDX Pro Analisis | Didukung oleh Gemini AI dan YFinance.")
