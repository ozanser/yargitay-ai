import streamlit as st
from PIL import Image, ImageEnhance
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import json
import time

# --- 1. AYARLAR ---
st.set_page_config(page_title="Yargıtay AI Asistanı", layout="wide", page_icon="⚖️")

# --- 2. GÜVENLİK KONTROLÜ (EN BAŞTA) ---

# Session State kontrolü (İlk kez mi açılıyor?)
if 'giris_yapildi' not in st.session_state:
    st.session_state['giris_yapildi'] = False

# --- GİRİŞ EKRANI TASARIMI ---
if not st.session_state['giris_yapildi']:
    st.markdown("## 🔒 Güvenli Yargıtay Sistemi")
    st.info("Lütfen devam etmek için erişim şifresini giriniz.")
    
    sifre = st.text_input("Şifre:", type="password")
    
    if st.button("Giriş Yap", type="primary"):
        try:
            # Şifre Kontrolü
            dogru_sifre = st.secrets["APP_PASSWORD"]
            if sifre == dogru_sifre:
                st.session_state['giris_yapildi'] = True
                st.success("Giriş Başarılı!")
                st.rerun() # Sayfayı yenile ve içeri al
            else:
                st.error("Hatalı Şifre!")
        except:
            # Eğer secrets ayarlanmadıysa test için "1234" kabul et
            if sifre == "1234":
                st.session_state['giris_yapildi'] = True
                st.rerun()
            else:
                st.error("Hatalı Şifre! (Secrets ayarlı değilse 1234 deneyin)")
    
    st.stop() # GİRİŞ YAPILMADIYSA KOD BURADA DURUR, AŞAĞI İNMEZ!

# ====================================================
# BURADAN AŞAĞISI SADECE GİRİŞ YAPANLARA GÖRÜNÜR
# ====================================================

# --- YAN MENÜ (ÇIKIŞ BUTONU BURADA) ---
with st.sidebar:
    st.success(f"✅ Oturum Açık")
    
    # ÇIKIŞ YAP BUTONU
    if st.button("🚪 Çıkış Yap", type="secondary"):
        st.session_state['giris_yapildi'] = False
        st.rerun() # Sayfayı yenile ve giriş ekranına dön
        
    st.markdown("---")
    st.header("⚙️ Yönetim Paneli")

# --- 3. BAĞLANTILAR VE FONKSİYONLAR ---

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    SUPABASE_URL = "URL"
    SUPABASE_KEY = "KEY"

@st.cache_resource
def init_supabase():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except:
        return None

supabase = init_supabase()

@st.cache_resource
def model_yukle():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = model_yukle()

# Veritabanı İstatistiklerini Yan Menüye Ekle
if supabase:
    try:
        with st.sidebar:
            sayi = supabase.table("kararlar").select("id", count="exact").execute().count
            st.metric("Kayıtlı Karar", sayi)
            
            if st.button("Kopyaları Temizle"):
                st.toast("Temizlik fonksiyonu çalışıyor...")
                # Temizlik kodu buraya eklenebilir
    except:
        pass

# --- 4. ANA FONKSİYONLAR ---

def ocr_isleme(image):
    img = image.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    try:
        return pytesseract.image_to_string(img, lang='tur')
    except:
        return pytesseract.image_to_string(img)

def veritabanina_kaydet(metin, vektor):
    if not supabase: return False
    vektor_json = json.dumps(vektor.tolist())
    data = {"metin": metin, "vektor": vektor_json}
    try:
        supabase.table("kararlar").insert(data).execute()
        return True
    except: return False

def mukerrer_kontrol(yeni_vektor):
    if not supabase: return False
    response = supabase.table("kararlar").select("vektor").execute()
    if not response.data: return False
    yeni_vektor_np = yeni_vektor.astype(np.float32)
    for satir in response.data:
        try:
            db_vektor = np.array(json.loads(satir['vektor'])).astype(np.float32)
            if util.cos_sim(yeni_vektor_np, db_vektor).item() > 0.90: return True
        except: continue
    return False

def arama_yap_hibrit(sorgu, esik_degeri):
    if not supabase: return []
    try:
        response = supabase.table("kararlar").select("*").execute()
        veriler = response.data
    except: return []
    if not veriler: return []

    sorgu_vektoru = model.encode(sorgu, convert_to_tensor=False).astype(np.float32)
    sonuclar = []
    sorgu_kucuk = sorgu.lower()

    for satir in veriler:
        try:
            db_vektor = np.array(json.loads(satir['vektor'])).astype(np.float32)
            vektor_skoru = util.cos_sim(sorgu_vektoru, db_vektor).item()
            bonus = 0.30 if sorgu_kucuk in satir['metin'].lower() else 0.0
            toplam = vektor_skoru + bonus
            
            if toplam >= esik_degeri:
                sonuclar.append({'metin': satir['metin'], 'skor': toplam, 'bonus': bonus})
        except: continue
    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 5. ANA EKRAN SEKMELERİ ---

st.title("⚖️ Yargıtay AI & OCR Sistemi")

tab1, tab2 = st.tabs(["📤 Karar Yükle", "🔍 Arama Yap"])

with tab1:
    files = st.file_uploader("Dosya Seç", accept_multiple_files=True)
    if files and st.button("Kaydet", type="primary"):
        bar = st.progress(0)
        basarili = 0
        for i, f in enumerate(files):
            try:
                img = Image.open(f)
                txt = ocr_isleme(img)
                if len(txt) > 10:
                    v = model.encode(txt, convert_to_tensor=False).astype(np.float32)
                    if not mukerrer_kontrol(v):
                        if veritabanina_kaydet(txt, v): basarili += 1
            except: pass
            bar.progress((i+1)/len(files))
        st.success(f"{basarili} dosya kaydedildi.")

with tab2:
    col1, col2 = st.columns([3,1])
    with col1: sorgu = st.text_input("Arama")
    with col2: esik = st.slider("Hassasiyet", 0.0, 1.0, 0.25)
    
    if st.button("Ara"):
        res = arama_yap_hibrit(sorgu, esik)
        if res:
            for r in res:
                st.info(f"Puan: %{int(r['skor']*100)} - {r['metin'][:200]}...")
        else:
            st.warning("Bulunamadı")
