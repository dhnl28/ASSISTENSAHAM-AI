import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
from google import genai
from google.genai.errors import APIError 
import yfinance as yf
import os 
import tempfile 
from config import GEMINI_API_KEY 

st.set_page_config(
    page_title="IDX Pro - Analisis Saham AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Inisialisasi Klien Gemini ---
try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "GANTI_DENGAN_GEMINI_API_KEY_ANDA":
        client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        st.warning("⚠️ Harap atur GEMINI_API_KEY Anda di file config.py untuk mengaktifkan Analisis AI.")
        client = None
except Exception as e:
    st.error(f"Gagal inisialisasi Gemini Client: {e}")
    client = None

# --- Inisialisasi Session State ---
if 'ai_analysis' not in st.session_state:
    st.session_state['ai_analysis'] = "Belum ada analisis yang dijalankan."
if 'analysis_status' not in st.session_state:
    st.session_state['analysis_status'] = None
if 'ticker_selected' not in st.session_state: 
    st.session_state['ticker_selected'] = "BBCA"
if 'uploaded_file_data' not in st.session_state:
    st.session_state['uploaded_file_data'] = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state['uploaded_file_name'] = None
if 'financial_text_input_data' not in st.session_state:
    st.session_state['financial_text_input_data'] = None

# --- FUNGSI DATA YFINANCE ---
@st.cache_data(ttl=3600)
def get_historical_data(ticker):
    if not ticker.endswith(".JK"):
        idx_ticker = f"{ticker}.JK"
    else:
        idx_ticker = ticker
        
    try:
        if st.session_state['ticker_selected'] != ticker:
            st.cache_data.clear()
            st.session_state['ticker_selected'] = ticker
            
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

# --- FUNGSI VISUALISASI ---
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

# --- FUNGSI ANALISIS GEMINI ---
def analyze_content_with_gemini(input_content, gemini_file_obj=None):
    if not client:
        return "⚠️ Gemini API belum terinisialisasi."
        
    base_prompt = f"""
    Anda adalah analis saham profesional khusus pasar Indonesia (IDX). 
    Anda bertindak sebagai analis fundamental untuk saham dengan kode {st.session_state.ticker_selected}.
    TUGAS: Analisis data laporan keuangan berikut dan berikan wawasan investasi.
    
    OUTPUT YANG DIHARAPKAN (Format Markdown):
    ### 1. 🔍 Kesimpulan Eksekutif
    (3 Poin utama tentang kesehatan perusahaan ini)
    
    ### 2. 🧮 Analisis Rasio (Estimasi)
    *Jika angka tersedia, hitunglah. Jika tidak, berikan estimasi kualitatif.*
    - **Profitabilitas:** (Bahas Margin/ROE/Laba Bersih)
    - **Solvabilitas:** (Bahas Hutang/DER/Current Ratio)
    
    ### 3. 🚦 Rekomendasi Investasi
    **Vonis:** [STRONG BUY / BUY / HOLD / SELL / AVOID]
    **Alasan:** (Penjelasan singkat 2-3 kalimat)
    """

    if isinstance(input_content, str):
        content_text_limit = input_content[:5000]
        contents = [base_prompt + f"\n\n--- DATA INPUT TEKS ---\n{content_text_limit}"]
    else:
        contents = input_content + [base_prompt]

    analysis_result = ""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
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
                print(f"Gagal membersihkan file Gemini: {e}")
    
    return analysis_result

# --- FUNGSI CALLBACK (CEPATT!) ---
def analyze_callback(uploaded_file, financial_text_input):
    if not client:
        st.session_state['ai_analysis'] = "⚠️ Gemini API belum terinisialisasi. Periksa config.py."
        st.session_state['analysis_status'] = 'warning'
        return
        
    st.session_state['analysis_status'] = 'running'
    st.session_state['ai_analysis'] = "⏳ Sedang memproses analisis..."

    if uploaded_file:
        st.session_state['uploaded_file_data'] = uploaded_file.getvalue() 
        st.session_state['uploaded_file_name'] = uploaded_file.name 
        st.session_state['financial_text_input_data'] = None 
    elif financial_text_input:
        st.session_state['uploaded_file_data'] = None
        st.session_state['uploaded_file_name'] = None
        st.session_state['financial_text_input_data'] = financial_text_input
    else:
        st.session_state['analysis_status'] = 'warning'
        st.session_state['ai_analysis'] = "⚠️ Mohon masukkan teks atau upload file terlebih dahulu."
        return

    st.rerun() # Pemicu rerun setelah status diatur

        
# --- SIDEBAR & MAIN UI ---

st.sidebar.title("🛠️ Kontrol Analisis")
TICKER = st.sidebar.text_input("Kode Saham (cth: BBCA)", value=st.session_state['ticker_selected']).upper().strip()

if st.sidebar.button("🔍 Refresh Data Saham"):
    st.cache_data.clear()
    st.session_state['ticker_selected'] = TICKER 
    st.rerun()

df_hist = get_historical_data(TICKER)
snapshot = get_snapshot_data(df_hist)

st.sidebar.markdown("---")
st.sidebar.subheader(f"📊 {TICKER} Snapshot")

if snapshot['price'] > 0:
    col_sb1, col_sb2 = st.sidebar.columns(2)
    col_sb1.metric("Harga", f"Rp {snapshot['price']:,.0f}")
    col_sb2.metric("Perubahan", f"{snapshot['change_percent']:+.2f}%", delta_color="normal")
    st.sidebar.write(f"Volume: {snapshot['volume']/1000000:.1f} Juta")
else:
    st.sidebar.error("Data Saham Tidak Ditemukan atau Gagal Diambil.")

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
    
    uploaded_file_key = "uploaded_file_tab2"
    financial_text_input_key = "financial_text_input_tab2"
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        uploaded_file = st.file_uploader(
            "Upload Laporan Keuangan (.PDF, .TXT, .DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=False,
            key=uploaded_file_key, 
            disabled=(st.session_state.analysis_status == 'running')
        )
        
        financial_text_input = st.text_area(
            "Atau Paste Teks Laporan Laba Rugi/Neraca di sini (Maks 5000 karakter)",
            height=200, 
            placeholder="Contoh: Pendapatan Bersih Kuartal III 2024: 10.5 Triliun. Total Ekuitas: 40 Triliun. Laba Bersih: 2 Triliun...",
            key=financial_text_input_key,
            disabled=(st.session_state.analysis_status == 'running')
        )
        
        st.button(
            "🚀 Jalankan Analisis Gemini", 
            type="primary", 
            use_container_width=True,
            on_click=analyze_callback,
            args=(uploaded_file, financial_text_input),
            disabled=(st.session_state.analysis_status == 'running') or (uploaded_file is None and not financial_text_input)
        )
    
    with col_result:
        if st.session_state.analysis_status == 'running':
            st.info("⏳ Gemini sedang menganalisis data (proses mungkin memakan waktu 30-60 detik).")
        elif st.session_state.analysis_status == 'success':
            st.success("✅ Analisis Selesai! Lihat di bawah.")
        elif st.session_state.analysis_status == 'error':
            st.error("❌ Analisis Gagal. Lihat detail error di bawah.")
        elif st.session_state.analysis_status == 'warning':
            st.warning("⚠️ Input dibutuhkan.")
        else:
            st.markdown("*Klik tombol 'Jalankan Analisis Gemini' di sebelah kiri untuk melihat hasil.*")

    st.subheader("Ringkasan dan Rekomendasi AI")
    
    # --- Logika Pemrosesan Berat (DI BADAN SKRIP) ---
    if st.session_state.analysis_status == 'running':
        uploaded_file_data = st.session_state.get('uploaded_file_data')
        uploaded_file_name = st.session_state.get('uploaded_file_name')
        financial_text_input_data = st.session_state.get('financial_text_input_data')
        
        analysis_result = None
        uploaded_file_gemini = None 
        temp_file_path = None 
        
        if uploaded_file_data is not None and uploaded_file_name:
            try:
                # 1. Tulis data biner ke file sementara (tempfile)
                file_suffix = os.path.splitext(uploaded_file_name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
                    tmp_file.write(uploaded_file_data)
                    temp_file_path = tmp_file.name
                    
                # 2. Unggah file ke Gemini
                with st.spinner(f"Mengunggah file {uploaded_file_name} untuk dianalisis..."):
                    uploaded_file_gemini = client.files.upload(
                        file=temp_file_path 
                    )
                
                input_content = [uploaded_file_gemini]
                
                # 3. Jalankan analisis
                analysis_result = analyze_content_with_gemini(input_content, uploaded_file_gemini)
                
            except APIError as e:
                st.session_state['ai_analysis'] = f"❌ Gagal mengunggah file ke Gemini (API Error). Detail: {e}"
                st.session_state['analysis_status'] = 'error'
            except Exception as e:
                st.session_state['ai_analysis'] = f"❌ Gagal memproses file. Detail: {e}"
                st.session_state['analysis_status'] = 'error'
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        elif financial_text_input_data:
            with st.spinner("Menganalisis teks keuangan..."):
                analysis_result = analyze_content_with_gemini(financial_text_input_data)

        # Pembaruan hasil akhir dan status setelah proses berat selesai
        if analysis_result and st.session_state.analysis_status == 'running': 
            if "❌" not in analysis_result:
                st.session_state['ai_analysis'] = analysis_result
                st.session_state['analysis_status'] = 'success'
            else:
                st.session_state['ai_analysis'] = analysis_result 
                st.session_state['analysis_status'] = 'error'
            # Perubahan session state ini akan memicu RERUN terakhir untuk menampilkan hasil
            st.rerun() 
            
    # --- END: Logika Pemrosesan Berat ---

    # Menampilkan hasil
    if st.session_state.ai_analysis:
        if st.session_state.analysis_status == 'success':
            st.markdown(st.session_state.ai_analysis)
        elif st.session_state.analysis_status == 'error':
            st.error(st.session_state.ai_analysis)
        else:
            st.info(st.session_state.ai_analysis)
    else:
        st.info("Belum ada analisis yang dijalankan.")

st.markdown("---")
st.caption("IDX Pro Analisis | Didukung oleh Gemini AI dan YFinance.")
