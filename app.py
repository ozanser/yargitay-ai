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

# --- 2. GÜVENLİK (GİRİŞ KONTROLÜ) ---
if 'giris_yapildi' not in st.session_state:
    st.session_state['giris_yapildi'] = False

if not st.session_state['giris_yapildi']:
    st.markdown("## 🔒 Güvenli Yargıtay Sistemi")
    
    gecerli_sifre = "1234" # Varsayılan
    try:
        if "APP_PASSWORD" in st.secrets:
            gecerli_sifre = st.secrets["APP_PASSWORD"]
    except: pass

    girilen = st.text_input("Şifre:", type="password")
    
    if st.button("Giriş", type="primary"):
        if girilen == gecerli_sifre:
            st.session_state['giris_yapildi'] = True
            st.success("Giriş Yapıldı")
            time.sleep(0.3)
            st.rerun()
        else:
            st.error("Hatalı Şifre")
            if gecerli_sifre == "1234": st.caption("İpucu: 1234")
    
    st.stop() # Giriş yoksa dur

# ==========================================
# İÇERİK (SADECE GİRİŞ YAPANLARA)
# ==========================================

# --- 3. BAĞLANTILAR ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    SUPABASE_URL = "URL"
    SUPABASE_KEY = "KEY"

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

# --- 4. TEMİZLİK VE YÖNETİM FONKSİYONLARI ---

def veritabani_temizle():
    """Kopya kayıtları veritabanından siler."""
    if not supabase: return 0
    # Sadece ID ve Metin çekiyoruz (Hız için)
    res = supabase.table("kararlar").select("id, metin").execute()
    if not res.data: return 0
    
    gordum = set()
    silinecek = []
    
    for s in res.data:
        # Metnin ilk 100 karakterini imza olarak kullan
        imza = s['metin'].strip()[:100]
        if imza in gordum:
            silinecek.append(s['id'])
        else:
            gordum.add(imza)
    
    if silinecek:
        supabase.table("kararlar").delete().in_("id", silinecek).execute()
        return len(silinecek)
    return 0

# --- 5. YAN MENÜ (YÖNETİM PANELİ) ---
with st.sidebar:
    st.success("✅ Yetkili Girişi")
    
    col_cikis, col_bos = st.columns([1,1])
    with col_cikis:
        if st.button("🚪 Çıkış"):
            st.session_state['giris_yapildi'] = False
            st.rerun()
            
    st.divider()
    st.header("⚙️ Yönetim Paneli")
    
    if supabase:
        try:
            # Kayıt Sayısını Göster
            sayi = supabase.table("kararlar").select("id", count="exact").execute().count
            st.metric("Toplam Karar", sayi)
            
            st.markdown("---")
            st.write("Veritabanı Bakımı")
            
            # --- GERİ GETİRİLEN BUTON BURADA ---
            if st.button("🧹 Kopyaları Temizle", type="primary"):
                with st.spinner("Taranıyor..."):
                    silinen = veritabani_temizle()
                    if silinen > 0:
                        st.success(f"{silinen} adet kopya silindi!")
                        time.sleep(2)
                        st.rerun() # Sayıyı güncellemek için yenile
                    else:
                        st.info("Veritabanı temiz.")
        except:
            st.error("Veritabanına bağlanılamadı")

# --- 6. İŞLEM FONKSİYONLARI ---

def ocr_isleme(image):
    img = image.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    try: return pytesseract.image_to_string(img, lang='tur')
    except: return pytesseract.image_to_string(img)

def veritabanina_kaydet(metin, vektor):
    if not supabase: return False
    v_json = json.dumps(vektor.tolist())
    data = {"metin": metin, "vektor": v_json}
    try:
        supabase.table("kararlar").insert(data).execute()
        return True
    except: return False

def mukerrer_kontrol(yeni_v):
    if not supabase: return False
    res = supabase.table("kararlar").select("vektor").execute()
    if not res.data: return False
    yeni_v = yeni_v.astype(np.float32)
    for row in res.data:
        try:
            db_v = np.array(json.loads(row['vektor'])).astype(np.float32)
            if util.cos_sim(yeni_v, db_v).item() > 0.90: return True
        except: continue
    return False

def arama_yap_hibrit(sorgu, esik):
    if not supabase: return []
    try: res = supabase.table("kararlar").select("*").execute()
    except: return []
    if not res.data: return []

    sorgu_v = model.encode(sorgu, convert_to_tensor=False).astype(np.float32)
    sorgu_lower = sorgu.lower()
    sonuclar = []

    for row in res.data:
        try:
            db_v = np.array(json.loads(row['vektor'])).astype(np.float32)
            skor = util.cos_sim(sorgu_v, db_v).item()
            bonus = 0.30 if sorgu_lower in row['metin'].lower() else 0.0
            total = skor + bonus
            if total >= esik:
                sonuclar.append({'metin': row['metin'], 'skor': total, 'bonus': bonus})
        except: continue
    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 7. ANA EKRAN SEKMELERİ ---

st.title("⚖️ Yargıtay AI & OCR Sistemi")

tab1, tab2 = st.tabs(["📤 Çoklu Yükleme", "🔍 Hassas Arama"])

with tab1:
    files = st.file_uploader("Dosya Seç", accept_multiple_files=True)
    if files and st.button("Kaydet", type="primary"):
        bar = st.progress(0)
        basarili = 0
        for i, f in enumerate(files):
            try:
                img = Image.open(f)
                txt = ocr_isleme(img)
                if len(txt) > 10:
                    v = model.encode(txt, convert_to_tensor=False).astype(np.float32)
                    if not mukerrer_kontrol(v):
                        if veritabanina_kaydet(txt, v): basarili += 1
            except: pass
            bar.progress((i+1)/len(files))
        st.success(f"{basarili} dosya kaydedildi.")

with tab2:
    c1, c2 = st.columns([3,1])
    with c1: q = st.text_input("Arama")
    with c2: tr = st.slider("Hassasiyet", 0.0, 1.0, 0.25)
    
    if st.button("Ara"):
        if q:
            with st.spinner("Aranıyor..."):
                res = arama_yap_hibrit(q, tr)
                if res:
                    for r in res:
                        st.info(f"Puan: %{int(r['skor']*100)} - {r['metin'][:300]}...")
                else: st.warning("Bulunamadı")
        else: st.warning("Kelime girin")
