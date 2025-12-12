import streamlit as st
from PIL import Image, ImageEnhance
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import json

# --- 1. AYARLAR ---
st.set_page_config(page_title="Yargıtay AI (Debug)", layout="wide", page_icon="🐞")

# --- 2. GÜVENLİK ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    st.error("Supabase sırları (Secrets) bulunamadı!")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

@st.cache_resource
def model_yukle():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = model_yukle()

# --- 3. FONKSİYONLAR ---

def ocr_isleme(image):
    # Görüntü iyileştirme
    img = image.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    try:
        text = pytesseract.image_to_string(img, lang='tur')
        return text
    except:
        return pytesseract.image_to_string(img)

def veritabanina_kaydet(metin, vektor):
    vektor_json = json.dumps(vektor.tolist())
    data = {"metin": metin, "vektor": vektor_json}
    try:
        supabase.table("kararlar").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Kayıt hatası detayı: {e}") # Hatayı göster
        return False

# --- GÜNCELLENMİŞ VE KONUŞKAN ARAMA FONKSİYONU ---
def arama_yap_debug(sorgu):
    st.info("1. Veritabanına bağlanılıyor...")
    
    try:
        # Tüm verileri çek
        response = supabase.table("kararlar").select("*").execute()
        veriler = response.data
    except Exception as e:
        st.error(f"Veritabanı okuma hatası: {e}")
        return []

    if not veriler:
        st.warning("Veritabanı BOŞ! Hiç kayıt dönmedi.")
        return []
    
    st.write(f"📂 Veritabanında {len(veriler)} adet kayıt bulundu. Analiz ediliyor...")

    # Sorguyu vektöre çevir
    try:
        sorgu_vektoru = model.encode(sorgu, convert_to_tensor=False)
    except Exception as e:
        st.error(f"Model hatası: {e}")
        return []

    sonuclar = []
    
    # Her satırı tek tek kontrol et ve ekrana yaz
    for i, satir in enumerate(veriler):
        try:
            # Vektör kontrolü
            if satir['vektor'] is None:
                st.warning(f"Satır {i}: Vektör verisi yok, atlanıyor.")
                continue

            db_vektor = np.array(json.loads(satir['vektor']))
            skor = util.cos_sim(sorgu_vektoru, db_vektor).item()
            
            # Debug için skoru yazdıralım (Geliştirme aşamasında)
            # st.caption(f"Kayıt {i} Skoru: {skor:.4f}") 
            
            # Eşiği test için %10'a (0.10) çektim. Neredeyse her şeyi gösterecek.
            if skor > 0.10: 
                sonuclar.append(satir | {'skor': skor})
        except Exception as e:
            st.error(f"Satır {i} işlenirken hata: {e}")
            continue

    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 4. ARAYÜZ ---

st.title("🐞 Yargıtay AI - Hata Ayıklama Modu")
st.warning("Bu modda sistem yaptığı her adımı ekrana yazar.")

tab1, tab2 = st.tabs(["📤 Karar Yükle", "🔍 Arama Yap"])

with tab1:
    uploaded_file = st.file_uploader("Resim Yükle", type=["jpg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, width=200)
        
        if st.button("Kaydet"):
            metin = ocr_isleme(img)
            if len(metin) > 5:
                st.write("Metin okundu, kaydediliyor...")
                vektor = model.encode(metin, convert_to_tensor=False)
                if veritabanina_kaydet(metin, vektor):
                    st.success("✅ Kayıt Başarılı!")
                else:
                    st.error("❌ Kaydedilemedi.")
            else:
                st.error("Metin okunamadı.")

with tab2:
    st.header("Arama Testi")
    
    # Mevcut kayıt sayısını kontrol et
    if st.button("📊 Veritabanı Durumunu Kontrol Et"):
        res = supabase.table("kararlar").select("id", count="exact").execute()
        st.info(f"Supabase'de şu an toplam {res.count} adet kayıt var.")

    sorgu = st.text_input("Ne arıyorsunuz?", placeholder="Örn: delil yetersizliği")
    
    if st.button("🔎 Detaylı Arama Yap"):
        if not sorgu:
            st.error("Lütfen bir kelime yazın.")
        else:
            sonuclar = arama_yap_debug(sorgu)
            
            if sonuclar:
                st.success(f"Toplam {len(sonuclar)} sonuç bulundu.")
                for s in sonuclar:
                    st.markdown("---")
                    st.markdown(f"**Skor:** %{int(s['skor']*100)}")
                    st.info(s['metin'])
            else:
                st.error("Sonuç bulunamadı (Eşik değerinin altında kalmış olabilir veya veritabanı boş).")
