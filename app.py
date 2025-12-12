import streamlit as st
from PIL import Image, ImageEnhance
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import json

# --- 1. AYARLAR ---
st.set_page_config(page_title="Yargıtay AI Asistanı", layout="wide", page_icon="⚖️")

# --- 2. GÜVENLİK VE BAĞLANTILAR ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    SUPABASE_URL = "URL_YOKSA_BURAYA_YAZ"
    SUPABASE_KEY = "KEY_YOKSA_BURAYA_YAZ"

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

# --- 3. FONKSİYONLAR ---

def resim_on_isleme(image):
    # Görüntü iyileştirme
    img = image.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    return img

def ocr_isleme(image):
    processed_image = resim_on_isleme(image)
    try:
        text = pytesseract.image_to_string(processed_image, lang='tur')
        return text, processed_image
    except:
        text = pytesseract.image_to_string(processed_image)
        return text, processed_image

def veritabani_temizle():
    """Kopya kayıtları siler."""
    if not supabase: return 0
    try:
        # 'created_at' yerine 'tarih' veya 'id' kontrolü
        response = supabase.table("kararlar").select("id, metin").execute()
        veriler = response.data
    except:
        return 0

    if not veriler: return 0

    gordum_kumesi = set()
    silinecek_idler = []

    for satir in veriler:
        metin_imzasi = satir['metin'].strip()[:50] # İlk 50 harfe baksa yeter (hız için)
        if metin_imzasi in gordum_kumesi:
            silinecek_idler.append(satir['id'])
        else:
            gordum_kumesi.add(metin_imzasi)

    if silinecek_idler:
        try:
            supabase.table("kararlar").delete().in_("id", silinecek_idler).execute()
            return len(silinecek_idler)
        except:
            return 0
    return 0

def mukerrer_kontrol(yeni_vektor):
    if not supabase: return False, None
    response = supabase.table("kararlar").select("metin, vektor").execute()
    if not response.data: return False, None

    yeni_vektor_np = yeni_vektor
    for satir in response.data:
        try:
            db_vektor = np.array(json.loads(satir['vektor']))
            skor = util.cos_sim(yeni_vektor_np, db_vektor).item()
            if skor > 0.90: return True, satir
        except: continue
    return False, None

def veritabanina_kaydet(metin, vektor):
    if not supabase: return False
    vektor_json = json.dumps(vektor.tolist())
    data = {"metin": metin, "vektor": vektor_json}
    try:
        supabase.table("kararlar").insert(data).execute()
        return True
    except: return False

def arama_yap(sorgu):
    if not supabase: return []
    response = supabase.table("kararlar").select("*").execute()
    if not response.data: return []

    sorgu_vektoru = model.encode(sorgu, convert_to_tensor=False)
    sonuclar = []
    for satir in response.data:
        try:
            db_vektor = np.array(json.loads(satir['vektor']))
            skor = util.cos_sim(sorgu_vektoru, db_vektor).item()
            if skor > 0.35:
                sonuclar.append(satir | {'skor': skor})
        except: continue
    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 4. ARAYÜZ (BU KISIM KESİNLİKLE GÖRÜNECEK) ---

st.title("⚖️ Yargıtay AI & OCR Sistemi")

# Yan Menü
with st.sidebar:
    st.header("Yönetim")
    if supabase:
        try:
            sayi = supabase.table("kararlar").select("id", count="exact").execute().count
            st.metric("Toplam Karar", sayi)
        except:
            st.warning("Veritabanı Okunamadı")
    
    if st.button("Kopyaları Temizle"):
        silinen = veritabani_temizle()
        if silinen > 0:
            st.success(f"{silinen} kopya silindi.")
        else:
            st.info("Temiz.")

# ANA EKRAN SEKMELERİ (Burada girinti hatası yapılmamalı)
tab1, tab2 = st.tabs(["📤 Karar Yükle", "🔍 Arşivde Ara"])

# SEKME 1: YÜKLEME
with tab1:
    uploaded_file = st.file_uploader("Resim Yükle", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, width=200)
        
        if st.button("Kaydet", type="primary"):
            with st.spinner("İşleniyor..."):
                metin, islenmis = ocr_isleme(img)
                if len(metin) > 10:
                    vektor = model.encode(metin, convert_to_tensor=False)
                    var_mi, _ = mukerrer_kontrol(vektor)
                    if var_mi:
                        st.error("Bu karar zaten kayıtlı!")
                    else:
                        veritabanina_kaydet(metin, vektor)
                        st.success("Kaydedildi!")
                        st.write(metin)
                else:
                    st.error("Yazı okunamadı.")

# SEKME 2: ARAMA
with tab2:
    sorgu = st.text_input("Ne arıyorsunuz?")
    if st.button("Ara"):
        sonuclar = arama_yap(sorgu)
        if sonuclar:
            for s in sonuclar:
                st.success(f"Skor: %{int(s['skor']*100)}")
                st.write(s['metin'])
                st.markdown("---")
        else:
            st.warning("Bulunamadı.")
