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

# --- 2. GÜVENLİK ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    SUPABASE_URL = "URL_BURAYA"
    SUPABASE_KEY = "KEY_BURAYA"

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

def ocr_isleme(image):
    img = image.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    try:
        text = pytesseract.image_to_string(img, lang='tur')
        return text
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
    # Sadece vektörleri çekiyoruz (hız için)
    response = supabase.table("kararlar").select("vektor").execute()
    if not response.data: return False

    yeni_vektor_np = yeni_vektor.astype(np.float32)

    for satir in response.data:
        try:
            db_vektor = np.array(json.loads(satir['vektor'])).astype(np.float32)
            skor = util.cos_sim(yeni_vektor_np, db_vektor).item()
            if skor > 0.90: return True
        except: continue
    return False

def arama_yap(sorgu):
    if not supabase: return []
    try:
        response = supabase.table("kararlar").select("*").execute()
        veriler = response.data
    except: return []
    if not veriler: return []

    sorgu_vektoru = model.encode(sorgu, convert_to_tensor=False).astype(np.float32)
    sonuclar = []
    for satir in veriler:
        try:
            db_vektor = np.array(json.loads(satir['vektor'])).astype(np.float32)
            skor = util.cos_sim(sorgu_vektoru, db_vektor).item()
            if skor > 0.25:
                sonuclar.append(satir | {'skor': skor})
        except: continue
    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

def veritabani_temizle():
    if not supabase: return 0
    response = supabase.table("kararlar").select("id, metin").execute()
    if not response.data: return 0
    gordum = set()
    silinecek = []
    for s in response.data:
        imza = s['metin'].strip()[:50]
        if imza in gordum: silinecek.append(s['id'])
        else: gordum.add(imza)
    if silinecek:
        supabase.table("kararlar").delete().in_("id", silinecek).execute()
        return len(silinecek)
    return 0

# --- 4. ARAYÜZ ---

st.title("⚖️ Yargıtay AI & OCR Sistemi")

with st.sidebar:
    st.header("Yönetim")
    if supabase:
        try:
            sayi = supabase.table("kararlar").select("id", count="exact").execute().count
            st.metric("Kayıtlı Karar", sayi)
        except:
            st.metric("Durum", "Bağlanamadı")
    if st.button("Kopyaları Temizle"):
        s = veritabani_temizle()
        if s: st.success(f"{s} kopya silindi.")
        else: st.info("Temiz.")

tab1, tab2 = st.tabs(["📤 Çoklu Karar Yükle", "🔍 Arama Yap"])

# --- ÇOKLU YÜKLEME SİSTEMİ ---
with tab1:
    st.info("Birden fazla resim seçebilirsiniz (Ctrl tuşuna basılı tutarak).")
    
    # 1. DEĞİŞİKLİK: accept_multiple_files=True
    uploaded_files = st.file_uploader("Karar Resimlerini Yükle", 
                                      type=["jpg", "png", "jpeg"], 
                                      accept_multiple_files=True)
    
    if uploaded_files:
        # Kaç dosya seçildiğini göster
        st.write(f"📂 Toplam {len(uploaded_files)} adet dosya seçildi.")
        
        if st.button("Hepsini Analiz Et ve Kaydet", type="primary"):
            
            # İlerleme Çubuğu (Progress Bar)
            progress_bar = st.progress(0)
            durum_metni = st.empty()
            
            basarili = 0
            mukerrer = 0
            hatali = 0
            
            # 2. DEĞİŞİKLİK: Dosyalar üzerinde döngü
            for i, uploaded_file in enumerate(uploaded_files):
                dosya_adi = uploaded_file.name
                durum_metni.text(f"İşleniyor: {dosya_adi}...")
                
                try:
                    img = Image.open(uploaded_file)
                    metin = ocr_isleme(img)
                    
                    if len(metin) > 10:
                        vektor = model.encode(metin, convert_to_tensor=False).astype(np.float32)
                        
                        if mukerrer_kontrol(vektor):
                            mukerrer += 1
                        else:
                            if veritabanina_kaydet(metin, vektor):
                                basarili += 1
                            else:
                                hatali += 1
                    else:
                        hatali += 1 # Metin okunamadı
                        
                except Exception as e:
                    hatali += 1
                
                # Çubuğu güncelle
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            durum_metni.empty()
            st.success("İşlem Tamamlandı!")
            
            # Sonuç Karnesi
            col1, col2, col3 = st.columns(3)
            col1.metric("✅ Başarılı", basarili)
            col2.metric("⛔ Zaten Vardı (Atlandı)", mukerrer)
            col3.metric("⚠️ Okunamadı/Hata", hatali)

with tab2:
    sorgu = st.text_input("Arama yapın")
    if st.button("Ara"):
        if not sorgu:
            st.warning("Lütfen bir şey yazın.")
        else:
            with st.spinner("Arşiv taranıyor..."):
                sonuclar = arama_yap(sorgu)
                if sonuclar:
                    st.success(f"🎯 {len(sonuclar)} sonuç bulundu.")
                    for s in sonuclar:
                        st.markdown("---")
                        st.subheader(f"Uygunluk: %{int(s['skor']*100)}")
                        st.info(s['metin'])
                else:
                    st.warning("Sonuç bulunamadı.")
