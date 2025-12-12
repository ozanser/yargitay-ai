import streamlit as st
from PIL import Image
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import json

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Yargıtay Akıllı Arşiv", layout="wide", page_icon="⚖️")

# --- 1. GÜVENLİ BAĞLANTILAR ---
# Gerçek projelerde şifreler koda yazılmaz. st.secrets'tan çekilir.
try:
    supa_url = st.secrets["SUPABASE_URL"]
    supa_key = st.secrets["SUPABASE_KEY"]
except:
    st.error("Veritabanı anahtarları bulunamadı! Lütfen Streamlit Secrets ayarlarını yapın.")
    st.stop()

@st.cache_resource
def init_db():
    return create_client(supa_url, supa_key)

supabase = init_db()

@st.cache_resource
def load_ai_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_ai_model()

# --- 2. FONKSİYONLAR ---

def ocr_isleme(image):
    """Görüntüden Türkçe metin okur."""
    try:
        text = pytesseract.image_to_string(image, lang='tur')
        return text
    except:
        # Dil paketi hatası olursa İngilizce dene
        return pytesseract.image_to_string(image)

def veritabanina_yaz(metin, vektor):
    """Veriyi buluta güvenli şekilde yazar."""
    vektor_json = json.dumps(vektor.tolist())
    data = {"metin": metin, "vektor": vektor_json}
    
    # Supabase'e yazma işlemi
    try:
        supabase.table("kararlar").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Kayıt Hatası: {e}")
        return False

def arama_motoru(sorgu_metni):
    """Buluttaki tüm verileri çeker ve vektör benzerliği hesaplar."""
    # Not: Milyonlarca veri olsaydı veritabanı tarafında (pgvector) arama yapardık.
    # Ancak binlerce veri için Python tarafında yapmak daha hızlı ve bedavadır.
    
    # Tüm veriyi çek
    response = supabase.table("kararlar").select("*").execute()
    db_verileri = response.data
    
    if not db_verileri:
        return []

    sorgu_vektoru = model.encode(sorgu_metni, convert_to_tensor=True)
    sonuclar = []

    for satir in db_verileri:
        # Kayıtlı vektörü JSON'dan geri çevir
        db_vektor = np.array(json.loads(satir['vektor']))
        
        # Matematiksel benzerlik hesabı (Cosine Similarity)
        skor = util.cos_sim(sorgu_vektoru, db_vektor).item()
        
        # %30'un altındaki benzerlikleri gösterme (Gürültüyü engelle)
        if skor > 0.30:
            sonuclar.append({
                "metin": satir['metin'],
                "tarih": satir['tarih'],
                "skor": skor
            })
            
    # Skora göre sırala (En yüksek en üstte)
    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 3. KULLANICI ARAYÜZÜ ---

st.title("⚖️ Yargıtay İçtihat & Karar Bankası")
st.markdown("---")

menu = st.sidebar.selectbox("Menü", ["Karar Yükle", "Akıllı Arama"])

if menu == "Karar Yükle":
    st.header("📄 Yeni Karar Ekleme")
    st.info("Yüklediğiniz fotoğraflar OCR ile taranır, yapay zeka ile anlamlandırılır ve buluta kaydedilir.")
    
    uploaded_file = st.file_uploader("Karar Fotoğrafı (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=400, caption="Önizleme")
        
        if st.button("Sisteme Kaydet", type="primary"):
            with st.status("İşlem yapılıyor...", expanded=True) as status:
                st.write("📝 Metin okunuyor (OCR)...")
                okunan_metin = ocr_isleme(image)
                
                if len(okunan_metin) > 50:
                    st.write("🧠 Yapay zeka vektör oluşturuyor...")
                    vektor = model.encode(okunan_metin)
                    
                    st.write("☁️ Buluta kaydediliyor...")
                    basari = veritabanina_yaz(okunan_metin, vektor)
                    
                    if basari:
                        status.update(label="İşlem Başarılı!", state="complete", expanded=False)
                        st.success("Karar başarıyla arşivlendi!")
                        with st.expander("Okunan Metni Gör"):
                            st.text(okunan_metin)
                else:
                    status.update(label="Hata", state="error")
                    st.error("Görüntüden anlamlı bir metin okunamadı. Lütfen daha net bir fotoğraf yükleyin.")

elif menu == "Akıllı Arama":
    st.header("🔍 İçerik Bazlı Arama")
    st.caption("Kelime eşleşmesi değil, anlam eşleşmesi yapılır. (Örn: 'İş kazası' yazsanız bile 'tazminat' geçen kararları bulabilir)")
    
    sorgu = st.text_input("Arama ifadesini girin:", placeholder="Örn: kıdem tazminatı faiz başlangıcı")
    
    if st.button("Ara"):
        with st.spinner("Arşiv taranıyor..."):
            sonuclar = arama_motoru(sorgu)
            
            if sonuclar:
                st.success(f"{len(sonuclar)} adet ilgili karar bulundu.")
                for i, res in enumerate(sonuclar[:10]): # İlk 10 sonuç
                    st.markdown(f"### {i+1}. Sonuç (Uygunluk: %{int(res['skor']*100)})")
                    st.caption(f"📅 Eklenme Tarihi: {res['tarih'][:10]}")
                    st.info(res['metin'][:600] + " ...[devamı var]")
                    st.divider()
            else:
                st.warning("Aradığınız kritere uygun karar bulunamadı.")