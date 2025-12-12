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
    SUPABASE_URL = "URL_YOKSA_BURAYA"
    SUPABASE_KEY = "KEY_YOKSA_BURAYA"

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

# --- GÜNCELLENEN HİBRİT ARAMA FONKSİYONU ---
def arama_yap_hibrit(sorgu, esik_degeri):
    if not supabase: return []
    try:
        response = supabase.table("kararlar").select("*").execute()
        veriler = response.data
    except: return []
    if not veriler: return []

    # 1. Vektör Araması (Anlam Araması)
    sorgu_vektoru = model.encode(sorgu, convert_to_tensor=False).astype(np.float32)
    
    sonuclar = []
    
    # Küçük harfe çevirip arama yapalım (Büyük/küçük harf duyarlılığını kaldırmak için)
    sorgu_kucuk = sorgu.lower()

    for satir in veriler:
        try:
            # A. Vektör Puanı Hesapla (0.0 - 1.0 arası)
            db_vektor = np.array(json.loads(satir['vektor'])).astype(np.float32)
            vektor_skoru = util.cos_sim(sorgu_vektoru, db_vektor).item()
            
            # B. Kelime Bonusu (Keyword Boosting)
            # Eğer aranan kelime metnin içinde birebir geçiyorsa puana +0.3 ekle!
            metin_kucuk = satir['metin'].lower()
            bonus_puan = 0.0
            
            if sorgu_kucuk in metin_kucuk:
                bonus_puan = 0.30  # Ciddi bir artış, o kararı tepeye taşır.
            
            # C. Toplam Skor
            # Bonus ile birlikte skor 1.0'ı geçebilir, sorun değil.
            toplam_skor = vektor_skoru + bonus_puan
            
            # D. Eşik Değeri Kontrolü (Kullanıcının seçtiği ayara göre)
            if toplam_skor >= esik_degeri:
                sonuclar.append({
                    'metin': satir['metin'], 
                    'skor': toplam_skor, 
                    'vektor_skoru': vektor_skoru, # Saf AI puanı
                    'bonus': bonus_puan           # Kelime eşleşmesi var mı?
                })

        except: continue
        
    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

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

tab1, tab2 = st.tabs(["📤 Çoklu Karar Yükle", "🔍 Hassas Arama"])

with tab1:
    st.info("Toplu yükleme yapabilirsiniz.")
    uploaded_files = st.file_uploader("Karar Resimlerini Yükle", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"📂 {len(uploaded_files)} dosya seçildi.")
        if st.button("Analiz Et ve Kaydet", type="primary"):
            progress_bar = st.progress(0)
            basarili, mukerrer, hatali = 0, 0, 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    img = Image.open(uploaded_file)
                    metin = ocr_isleme(img)
                    if len(metin) > 10:
                        vektor = model.encode(metin, convert_to_tensor=False).astype(np.float32)
                        if mukerrer_kontrol(vektor): mukerrer += 1
                        else:
                            if veritabanina_kaydet(metin, vektor): basarili += 1
                            else: hatali += 1
                    else: hatali += 1
                except: hatali += 1
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success("Bitti!")
            c1, c2, c3 = st.columns(3)
            c1.metric("✅ Başarılı", basarili)
            c2.metric("⛔ Mükerrer", mukerrer)
            c3.metric("⚠️ Hata", hatali)

# --- YENİLENEN ARAMA ARAYÜZÜ ---
with tab2:
    col_arama, col_ayar = st.columns([3, 1])
    
    with col_arama:
        sorgu = st.text_input("Aranacak kelime veya konu:", placeholder="Örn: eroin ticareti")
    
    with col_ayar:
        # Hassasiyet Ayarı (Slider)
        # Düşük (0.1): Her şeyi getirir (Alakasızlar dahil).
        # Yüksek (0.6): Sadece çok kesin olanları getirir.
        esik = st.slider("Hassasiyet", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
        st.caption("Sağa çekerseniz sadece kesin sonuçlar gelir.")

    if st.button("Ara"):
        if not sorgu:
            st.warning("Lütfen bir şey yazın.")
        else:
            with st.spinner("Hibrit arama yapılıyor (Kelime + Anlam)..."):
                sonuclar = arama_yap_hibrit(sorgu, esik)
                
                if sonuclar:
                    st.success(f"🎯 {len(sonuclar)} sonuç bulundu.")
                    
                    for s in sonuclar:
                        st.markdown("---")
                        
                        # Skor Rozetleri
                        c1, c2 = st.columns([1, 4])
                        with c1:
                            st.metric("Toplam Puan", f"%{int(s['skor']*100)}")
                            
                            # Eğer kelime bonusu almışsa belirtelim
                            if s['bonus'] > 0:
                                st.success("✅ Kelime Eşleşti!")
                            else:
                                st.info("🧠 Anlamsal Yakınlık")
                                
                        with c2:
                            st.info(s['metin'])
                else:
                    st.warning("Sonuç bulunamadı. Hassasiyeti düşürüp (sola çekip) tekrar deneyin.")
