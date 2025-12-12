import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
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
    # Local test
    SUPABASE_URL = "URL_YOKSA_BURAYA_YAZ"
    SUPABASE_KEY = "KEY_YOKSA_BURAYA_YAZ"

@st.cache_resource
def init_supabase():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        return None

supabase = init_supabase()

@st.cache_resource
def model_yukle():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = model_yukle()

# --- 3. FONKSİYONLAR ---

def resim_on_isleme(image):
    img = image.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    enhancer_sharp = ImageEnhance.Sharpness(img)
    img = enhancer_sharp.enhance(1.5)
    return img

def ocr_isleme(image):
    processed_image = resim_on_isleme(image)
    try:
        text = pytesseract.image_to_string(processed_image, lang='tur')
        return text, processed_image
    except:
        text = pytesseract.image_to_string(processed_image)
        return text, processed_image

def mukerrer_kontrol(yeni_vektor):
    """Yeni yüklenen karar veritabanında var mı diye bakar."""
    if not supabase: return False, None
    
    # Sadece gerekli sütunları çekiyoruz
    response = supabase.table("kararlar").select("metin, vektor").execute()
    db_verileri = response.data
    
    if not db_verileri: return False, None

    yeni_vektor_np = yeni_vektor

    for satir in db_verileri:
        try:
            db_vektor = np.array(json.loads(satir['vektor']))
            skor = util.cos_sim(yeni_vektor_np, db_vektor).item()
            if skor > 0.90: # %90 Benzerlik Eşiği
                return True, satir 
        except: continue  
    return False, None

# --- DÜZELTİLEN FONKSİYON ---
def veritabani_temizle():
    """Veritabanındaki aynı içeriğe sahip kopyaları siler."""
    if not supabase: return 0
    
    # HATA BURADAYDI: 'created_at' yerine 'tarih' yazdık.
    # Tablondaki sütun ismi 'tarih' olduğu için onu kullanmalıyız.
    try:
        response = supabase.table("kararlar").select("id, metin, tarih").order("tarih").execute()
        veriler = response.data
    except Exception as e:
        st.error(f"Veri çekme hatası: {e}. Lütfen Supabase tablonuzda 'tarih' sütunu olduğundan emin olun.")
        return 0

    if not veriler: return 0

    gordum_kumesi = set()
    silinecek_idler = []

    for satir in veriler:
        # Metnin tamamını imza olarak kullan (Boşlukları temizle)
        metin_imzasi = satir['metin'].strip()

        if metin_imzasi in gordum_kumesi:
            # Bu metni daha önce gördük, demek ki bu bir kopya -> ID'yi not et
            silinecek_idler.append(satir['id'])
        else:
            # İlk kez görüyoruz -> Kaydet
            gordum_kumesi.add(metin_imzasi)

    # 2. Toplu Silme İşlemi
    if silinecek_idler:
        try:
            # Supabase'de "in_" komutu ile listedeki tüm ID'leri siler
            supabase.table("kararlar").delete().in_("id", silinecek_idler).execute()
            return len(silinecek_idler)
        except Exception as e:
            st.error(f"Silme hatası: {e}")
            return 0
    else:
        return 0

def veritabanina_kaydet(metin, vektor):
    if not supabase: return False
    vektor_json = json.dumps(vektor.tolist())
    # Supabase 'tarih' sütununu now() ile otomatik doldurur, göndermeye gerek yok.
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
                # Burada da created_at yerine tarih kullandık
                sonuclar.append({'metin': satir['metin'], 'skor': skor, 'tarih': satir.get('tarih', '')})
        except: continue

    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 4. ARAYÜZ ---

st.title("⚖️ Yargıtay AI & OCR Sistemi")

# --- YAN MENÜ ---
with st.sidebar:
    st.header("⚙️ Yönetim Paneli")
    
    if supabase:
        try:
            toplam_kayit = supabase.table("kararlar").select("id", count="exact").execute().count
            st.metric("Toplam Karar Sayısı", toplam_kayit)
        except:
            st.metric("Durum", "Bağlantı Yok")

    st.markdown("---")
