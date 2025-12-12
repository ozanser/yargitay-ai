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
    response = supabase.table("kararlar").select("*").execute()
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

# --- YENİ EKLENEN OTOMATİK TEMİZLİK FONKSİYONU ---
def veritabani_temizle():
    """Veritabanındaki aynı içeriğe sahip kopyaları siler."""
    if not supabase: return 0
    
    # 1. Tüm verileri 'Oluşturulma Tarihine' göre çek (Eskiler kalsın, yeniler silinsin)
    response = supabase.table("kararlar").select("id, metin, created_at").order("created_at").execute()
    veriler = response.data

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
                sonuclar.append({'metin': satir['metin'], 'skor': skor, 'tarih': satir.get('created_at', '')})
        except: continue

    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 4. ARAYÜZ ---

st.title("⚖️ Yargıtay AI & OCR Sistemi")

# --- GÜNCELLENEN YAN MENÜ ---
with st.sidebar:
    st.header("⚙️ Yönetim Paneli")
    
    # Veritabanı durumunu göster
    if supabase:
        try:
            toplam_kayit = supabase.table("kararlar").select("id", count="exact").execute().count
            st.metric("Toplam Karar Sayısı", toplam_kayit)
        except:
            st.metric("Durum", "Bağlantı Yok")

    st.markdown("---")
    st.write("Veritabanı Bakımı")
    
    if st.button("🧹 Mükerrerleri Temizle", type="primary"):
        with st.spinner("Veritabanı taranıyor ve temizleniyor..."):
            silinen_sayisi = veritabani_temizle()
            
            if silinen_sayisi > 0:
                st.success(f"Toplam {silinen_sayisi} adet kopya kayıt silindi!")
                st.balloons() # Kutlama efekti :)
            else:
                st.info("Veritabanı tertemiz! Kopya kayıt bulunamadı.")

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
                st.write("🖼️ Görüntü işleniyor...")
                okunan_metin, islenmis_resim = ocr_isleme(original_image)
                with col2:
                    st.image(islenmis_resim, caption="İşlenmiş", width=300)

                if len(okunan_metin.strip()) > 20:
                    st.write("🔍 Benzerlik kontrolü yapılıyor...")
                    vektor = model.encode(okunan_metin, convert_to_tensor=False)
                    var_mi, eski_kayit = mukerrer_kontrol(vektor)
                    
                    if var_mi:
                        status.update(label="Mükerrer Kayıt Engellendi", state="error", expanded=True)
                        st.error("⛔ Bu karar zaten var!")
                        st.warning(f"Benzer kayıt içeriği: {eski_kayit['metin'][:100]}...")
                    else:
                        st.write("☁️ Kaydediliyor...")
                        if veritabanina_kaydet(okunan_metin, vektor):
                            status.update(label="Başarılı!", state="complete")
                            st.success("✅ Kaydedildi.")
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
                        st.caption(f"Tarih: {res['tarih'][:10]}")
                        st.info(res['metin'])
                else:
                    st.warning("Sonuç yok.")
