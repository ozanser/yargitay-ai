import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import json

# --- 1. AYARLAR VE KURULUM ---
st.set_page_config(page_title="Yargıtay AI Asistanı", layout="wide", page_icon="⚖️")

# --- 2. GÜVENLİK VE BAĞLANTILAR ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    # Local test için (Secrets yoksa)
    SUPABASE_URL = "SENIN_SUPABASE_URL_ADRESIN"
    SUPABASE_KEY = "SENIN_SUPABASE_ANON_KEY_ANAHTARIN"

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

# --- 3. GÖRÜNTÜ İŞLEME ---

def resim_on_isleme(image):
    """Görüntüyü gri yapar ve kontrastı artırır."""
    img = image.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    enhancer_sharp = ImageEnhance.Sharpness(img)
    img = enhancer_sharp.enhance(1.5)
    return img

def ocr_isleme(image):
    """İşlenmiş görüntüden metin okur."""
    processed_image = resim_on_isleme(image)
    try:
        text = pytesseract.image_to_string(processed_image, lang='tur')
        return text, processed_image
    except:
        text = pytesseract.image_to_string(processed_image)
        return text, processed_image

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

# --- 4. DÜZELTİLEN FONKSİYON (HATAYI ÇÖZEN KISIM) ---
def arama_yap(sorgu):
    if not supabase:
        return []
        
    response = supabase.table("kararlar").select("*").execute()
    db_verileri = response.data
    
    if not db_verileri:
        return []

    # DÜZELTME: convert_to_tensor=False yaptık.
    # Artık bu da bir Numpy Array oldu, veritabanı verisiyle uyumlu.
    sorgu_vektoru = model.encode(sorgu, convert_to_tensor=False)
    
    sonuclar = []

    for satir in db_verileri:
        try:
            # Veritabanından gelen veri (Numpy Array)
            db_vektor = np.array(json.loads(satir['vektor']))
            
            # İkisi de Numpy Array olduğu için artık hata vermez
            skor = util.cos_sim(sorgu_vektoru, db_vektor).item()
            
            if skor > 0.35:
                sonuclar.append({'metin': satir['metin'], 'skor': skor, 'tarih': satir.get('created_at', '')})
        except Exception as e:
            continue # Hatalı bir kayıt varsa atla, programı çökertme

    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 5. ARAYÜZ ---

st.title("⚖️ Yargıtay AI & OCR Sistemi")
st.markdown("**Gelişmiş Görüntü İşleme Modülü Devrede**")

tab1, tab2 = st.tabs(["📤 Karar Yükle", "🔍 Arşivde Ara"])

with tab1:
    st.header("Karar Fotoğrafı Yükle")
    uploaded_file = st.file_uploader("Görüntü seç (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        original_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Orjinal Görüntü", width=300)
        
        if st.button("Analiz Et ve Kaydet", type="primary"):
            with st.status("Görüntü işleniyor...", expanded=True) as status:
                st.write("🖼️ Görüntü temizleniyor...")
                okunan_metin, islenmis_resim = ocr_isleme(original_image)
                
                with col2:
                    st.image(islenmis_resim, caption="Bilgisayarın Gördüğü", width=300)

                if len(okunan_metin.strip()) > 20:
                    st.write("📝 Metin okundu.")
                    st.code(okunan_metin)
                    
                    st.write("🧠 Yapay zeka işliyor...")
                    # Kayıtta Numpy kullanmaya devam ediyoruz
                    vektor = model.encode(okunan_metin, convert_to_tensor=False)
                    
                    st.write("☁️ Kaydediliyor...")
                    basari = veritabanina_kaydet(okunan_metin, vektor)
                    
                    if basari:
                        status.update(label="Başarılı!", state="complete", expanded=False)
                        st.success("✅ Kaydedildi.")
                    else:
                        status.update(label="Hata", state="error")
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
