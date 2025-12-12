import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import json

# --- 1. AYARLAR VE KURULUM ---
st.set_page_config(page_title="Yargıtay AI Asistanı", layout="wide", page_icon="⚖️")

# Windows kullanıcıları için Tesseract yolu (Eğer sunucuda çalışıyorsa bu satırı yorum yapabilirsin)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- 2. GÜVENLİK VE BAĞLANTILAR ---
# Not: GitHub'a yüklerken şifreleri buraya yazma, Streamlit Secrets kullan!
# Local test için geçici olarak buraya yazabilirsin.
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    # Eğer secrets yoksa (bilgisayarında test ediyorsan) burayı doldur:
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

# --- 3. KRİTİK BÖLÜM: GÖRÜNTÜ İYİLEŞTİRME ---

def resim_on_isleme(image):
    """
    Renkli ve karmaşık arka planlı resimleri OCR için hazırlar.
    Resmi gri yapar ve kontrastı artırarak yazıları ortaya çıkarır.
    """
    # 1. Gri tona çevir (Siyah-Beyaz)
    img = image.convert('L')
    
    # 2. Kontrastı artır (Yazıyı arka plandan ayır)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Kontrastı 2 katına çıkar
    
    # 3. (Opsiyonel) Keskinleştirme
    enhancer_sharp = ImageEnhance.Sharpness(img)
    img = enhancer_sharp.enhance(1.5)
    
    return img

def ocr_isleme(image):
    """İşlenmiş görüntüden metin okur."""
    processed_image = resim_on_isleme(image)
    try:
        # Türkçe dil desteği ile oku
        text = pytesseract.image_to_string(processed_image, lang='tur')
        return text, processed_image
    except:
        # Hata olursa veya dil paketi yoksa varsayılanı dene
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

def arama_yap(sorgu):
    if not supabase:
        return []
        
    response = supabase.table("kararlar").select("*").execute()
    db_verileri = response.data
    
    if not db_verileri:
        return []

    sorgu_vektoru = model.encode(sorgu, convert_to_tensor=True)
    sonuclar = []

    for satir in db_verileri:
        db_vektor = np.array(json.loads(satir['vektor']))
        skor = util.cos_sim(sorgu_vektoru, db_vektor).item()
        
        # Skor %35'in üzerindeyse göster (Gürültüyü engelle)
        if skor > 0.35:
            sonuclar.append({'metin': satir['metin'], 'skor': skor, 'tarih': satir.get('created_at', '')})

    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 4. ARAYÜZ (FRONTEND) ---

st.title("⚖️ Yargıtay AI & OCR Sistemi")
st.markdown("**Gelişmiş Görüntü İşleme Modülü Devrede:** Karmaşık arka planlı kararları okuyabilir.")

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
                
                # 1. OCR İşlemi
                st.write("🖼️ Görüntü temizleniyor ve kontrast ayarlanıyor...")
                okunan_metin, islenmis_resim = ocr_isleme(original_image)
                
                # İşlenmiş resmi kullanıcıya gösterelim (Kanıt)
                with col2:
                    st.image(islenmis_resim, caption="Bilgisayarın Gördüğü (İşlenmiş)", width=300)

                # 2. Sonuç Kontrolü
                if len(okunan_metin.strip()) > 20:
                    st.write("📝 Metin başarıyla okundu.")
                    st.code(okunan_metin) # Okunan metni göster
                    
                    # 3. Vektör ve Kayıt
                    st.write("🧠 Yapay zeka anlamlandırıyor...")
                    vektor = model.encode(okunan_metin)
                    
                    st.write("☁️ Buluta kaydediliyor...")
                    basari = veritabanina_kaydet(okunan_metin, vektor)
                    
                    if basari:
                        status.update(label="İşlem Başarıyla Tamamlandı!", state="complete", expanded=False)
                        st.success("✅ Karar veritabanına güvenle eklendi.")
                    else:
                        status.update(label="Veritabanı Hatası", state="error")
                else:
                    status.update(label="Okuma Başarısız", state="error")
                    st.error("⚠️ Resimden anlamlı bir yazı çıkarılamadı.")
                    st.warning("İpucu: 'Bilgisayarın Gördüğü' resim simsiyah veya bembeyaz ise kontrast ayarı gerekebilir.")

with tab2:
    st.header("Akıllı Arama Motoru")
    arama_metni = st.text_input("Hukuki konu, kanun maddesi veya anahtar kelime:")
    
    if st.button("Araştır"):
        if not arama_metni:
            st.warning("Lütfen bir arama terimi girin.")
        else:
            with st.spinner("Veritabanı taranıyor..."):
                sonuclar = arama_yap(arama_metni)
                
                if sonuclar:
                    st.success(f"🎯 {len(sonuclar)} adet ilgili karar bulundu.")
                    for i, res in enumerate(sonuclar):
                        st.markdown("---")
                        st.subheader(f"{i+1}. Sonuç (Uygunluk: %{int(res['skor']*100)})")
                        st.info(res['metin'])
                else:
                    st.warning("😔 Aradığınız kritere uygun karar bulunamadı.")
