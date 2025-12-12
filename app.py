import streamlit as st
from PIL import Image, ImageEnhance
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import json

# --- 1. AYARLAR ---
st.set_page_config(page_title="Yargıtay AI Asistanı", layout="wide", page_icon="⚖️")

# --- 2. GÜVENLİK ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    # Yerel test için (Secrets yoksa burayı doldurabilirsin)
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
    # Görüntü netleştirme
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
    # Kaydederken de standart float listesi olarak kaydedelim
    vektor_json = json.dumps(vektor.tolist())
    data = {"metin": metin, "vektor": vektor_json}
    try:
        supabase.table("kararlar").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Kayıt Hatası: {e}")
        return False

def mukerrer_kontrol(yeni_vektor):
    if not supabase: return False
    response = supabase.table("kararlar").select("vektor").execute()
    if not response.data: return False

    # FIX: Yeni vektörü float32 yapıyoruz
    yeni_vektor_np = yeni_vektor.astype(np.float32)

    for satir in response.data:
        try:
            # FIX: Veritabanından geleni de float32 yapıyoruz
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

    # FIX 1: Sorgu vektörünü float32'ye zorluyoruz
    sorgu_vektoru = model.encode(sorgu, convert_to_tensor=False).astype(np.float32)
    
    sonuclar = []
    for satir in veriler:
        try:
            # FIX 2: Veritabanı vektörünü float32'ye zorluyoruz
            # Bu satır 'float != double' hatasını çözer
            db_vektor = np.array(json.loads(satir['vektor'])).astype(np.float32)
            
            skor = util.cos_sim(sorgu_vektoru, db_vektor).item()
            
            # Eşik değeri %25 (Biraz daha esnek olsun)
            if skor > 0.25:
                sonuclar.append(satir | {'skor': skor})
        except Exception as e:
            # Hatalı satırı atla ama durma
            continue

    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

def veritabani_temizle():
    """Kopya kayıtları temizler"""
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

# Yan Menü
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

# Ana Sekmeler
tab1, tab2 = st.tabs(["📤 Karar Yükle", "🔍 Arama Yap"])

with tab1:
    uploaded_file = st.file_uploader("Karar Resmi Yükle", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, width=250)
        
        if st.button("Kaydet", type="primary"):
            with st.spinner("Okunuyor..."):
                metin = ocr_isleme(img)
                if len(metin) > 10:
                    # Vektör oluştururken float32 yapıyoruz
                    vektor = model.encode(metin, convert_to_tensor=False).astype(np.float32)
                    
                    if mukerrer_kontrol(vektor):
                        st.error("⛔ Bu karar zaten var!")
                    else:
                        if veritabanina_kaydet(metin, vektor):
                            st.success("✅ Başarıyla Kaydedildi!")
                            with st.expander("Metni Gör"):
                                st.write(metin)
                else:
                    st.error("⚠️ Yazı okunamadı.")

with tab2:
    sorgu = st.text_input("Arama yapın (Örn: delil yetersizliği beraat)")
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
                    st.warning("😔 Sonuç bulunamadı.")
