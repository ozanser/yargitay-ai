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
    SUPABASE_URL = "SENIN_URL"
    SUPABASE_KEY = "SENIN_KEY"

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

# --- 3. YARDIMCI FONKSİYONLAR ---

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
    """
    Veritabanındaki tüm kararları kontrol eder.
    Eğer %90 üzeri benzerlik bulursa True (Var) döner.
    """
    response = supabase.table("kararlar").select("*").execute()
    db_verileri = response.data
    
    if not db_verileri:
        return False, None

    # Yeni vektörü tensör yapma, numpy kalsın
    yeni_vektor_np = yeni_vektor

    for satir in db_verileri:
        try:
            db_vektor = np.array(json.loads(satir['vektor']))
            skor = util.cos_sim(yeni_vektor_np, db_vektor).item()
            
            # %90 Benzerlik Eşiği (OCR hatalarını tolere etmek için %95 yerine %90 iyidir)
            if skor > 0.90:
                return True, satir # Mükerrer bulundu, bulunan kaydı döndür
        except:
            continue
            
    return False, None

def veritabanina_kaydet(metin, vektor):
    if not supabase:
        st.error("Veritabanı bağlantısı yok!")
        return False
        
    vektor_json = json.dumps(vektor.tolist())
    data = {"metin": metin, "vektor": vektor_json}
    
    try:
        supabase.table("kararlar").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Kayıt Hatası: {e}")
        return False

def arama_yap(sorgu):
    if not supabase: return []
    response = supabase.table("kararlar").select("*").execute()
    db_verileri = response.data
    if not db_verileri: return []

    sorgu_vektoru = model.encode(sorgu, convert_to_tensor=False)
    sonuclar = []

    for satir in db_verileri:
        try:
            db_vektor = np.array(json.loads(satir['vektor']))
            skor = util.cos_sim(sorgu_vektoru, db_vektor).item()
            if skor > 0.35:
                sonuclar.append({'metin': satir['metin'], 'skor': skor, 'tarih': satir.get('created_at', '')})
        except: continue

    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 4. ARAYÜZ ---

st.title("⚖️ Yargıtay AI & OCR Sistemi")

# --- YENİ ÖZELLİK: YAN MENÜ ---
with st.sidebar:
    st.header("⚙️ Yönetim Paneli")
    st.info("Veritabanı durumunu buradan kontrol edebilirsiniz.")
    
    if st.button("🧹 Mükerrer Kayıtları Temizle"):
        # Basit bir temizlik mantığı: Aynı metne sahip olanları siler
        st.warning("Bu işlem henüz otomatikleştirilmedi. Şu an için manuel kontrol önerilir.")
        # İleride buraya otomatik silme kodu ekleyebiliriz.

tab1, tab2 = st.tabs(["📤 Karar Yükle", "🔍 Arşivde Ara"])

with tab1:
    st.header("Karar Fotoğrafı Yükle")
    uploaded_file = st.file_uploader("Görüntü seç (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        original_image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Orjinal", width=300)
        
        if st.button("Analiz Et ve Kaydet", type="primary"):
            with st.status("İşlemler yapılıyor...", expanded=True) as status:
                
                # 1. OCR
                st.write("🖼️ Görüntü işleniyor...")
                okunan_metin, islenmis_resim = ocr_isleme(original_image)
                with col2:
                    st.image(islenmis_resim, caption="İşlenmiş", width=300)

                if len(okunan_metin.strip()) > 20:
                    st.write("📝 Metin Vektörleştiriliyor...")
                    vektor = model.encode(okunan_metin, convert_to_tensor=False)
                    
                    # 2. MÜKERRER KONTROLÜ (YENİ)
                    st.write("🔍 Benzerlik kontrolü yapılıyor...")
                    var_mi, eski_kayit = mukerrer_kontrol(vektor)
                    
                    if var_mi:
                        status.update(label="Kayıt Başarısız: Mükerrer!", state="error", expanded=True)
                        st.error("⛔ Bu karar zaten sistemde kayıtlı!")
                        st.warning(f"Sistemdeki benzer kayıt: \n\n {eski_kayit['metin'][:100]}...")
                    else:
                        st.write("☁️ Kaydediliyor...")
                        basari = veritabanina_kaydet(okunan_metin, vektor)
                        if basari:
                            status.update(label="Kaydedildi", state="complete")
                            st.success("✅ Karar başarıyla eklendi.")
                else:
                    status.update(label="Okunamadı", state="error")
                    st.error("⚠️ Yazı okunamadı.")

with tab2:
    st.header("Akıllı Arama")
    arama_metni = st.text_input("Arama terimi girin:")
    if st.button("Araştır"):
        if not arama_metni:
            st.warning("Bir şeyler yazın.")
        else:
            with st.spinner("Taranıyor..."):
                sonuclar = arama_yap(arama_metni)
                if sonuclar:
                    st.success(f"🎯 {len(sonuclar)} sonuç bulundu.")
                    for i, res in enumerate(sonuclar):
                        st.markdown("---")
                        st.subheader(f"{i+1}. Skor: %{int(res['skor']*100)}")
                        st.info(res['metin'])
                else:
                    st.warning("Sonuç yok.")
