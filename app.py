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

# --- 2. GÜVENLİK (KARARLI SÜRÜM) ---

# Session State Başlatma
if 'giris_yapildi' not in st.session_state:
    st.session_state['giris_yapildi'] = False

# Eğer giriş yapılmadıysa login ekranını göster
if not st.session_state['giris_yapildi']:
    st.markdown("## 🔒 Güvenli Yargıtay Sistemi")
    
    # Şifreyi Belirle (Hata yapısını kaldırdık, düz mantık)
    gecerli_sifre = "1234" # Varsayılan yedek şifre
    sifre_kaynagi = "Yedek (1234)"
    
    # Secrets kontrolü (Varsa oradan al, yoksa 1234 kalır)
    try:
        if "APP_PASSWORD" in st.secrets:
            gecerli_sifre = st.secrets["APP_PASSWORD"]
            sifre_kaynagi = "Sistem (Secrets)"
    except:
        pass # Secrets yoksa hata verme, devam et

    # Bilgilendirme (Sadece sen gör diye, production'da kaldırılabilir)
    # st.caption(f"Debug: Şifre kaynağı: {sifre_kaynagi}") 

    girilen_sifre = st.text_input("Erişim Şifresi:", type="password")
    
    if st.button("Giriş Yap", type="primary"):
        if girilen_sifre == gecerli_sifre:
            st.session_state['giris_yapildi'] = True
            st.success("Giriş Onaylandı!")
            time.sleep(0.5) # Kullanıcının mesajı görmesi için minik bir bekleme
            st.rerun()
        else:
            st.error("⛔ Hatalı Şifre!")
            if sifre_kaynagi == "Yedek (1234)":
                st.warning("İpucu: Sistemde özel şifre ayarlı değil, '1234' deneyin.")
    
    st.stop() # Kodun geri kalanını kesinlikle çalıştırma

# ====================================================
# BURASI SADECE GİRİŞ YAPANLARA AÇIK ALAN
# ====================================================

# --- YAN MENÜ ---
with st.sidebar:
    st.success("✅ Yetkili Girişi")
    if st.button("🚪 Çıkış Yap"):
        st.session_state['giris_yapildi'] = False
        st.rerun()
    st.divider()
    st.header("⚙️ Yönetim")

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

# İstatistik Gösterimi
if supabase:
    try:
        res = supabase.table("kararlar").select("id", count="exact").execute()
        with st.sidebar:
            st.metric("Toplam Karar", res.count)
    except:
        pass

# --- 4. FONKSİYONLAR ---

def ocr_isleme(image):
    img = image.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    try:
        return pytesseract.image_to_string(img, lang='tur')
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
    res = supabase.table("kararlar").select("vektor").execute()
    if not res.data: return False
    yeni_v = yeni_vektor.astype(np.float32)
    for row in res.data:
        try:
            db_v = np.array(json.loads(row['vektor'])).astype(np.float32)
            if util.cos_sim(yeni_v, db_v).item() > 0.90: return True
        except: continue
    return False

def arama_yap_hibrit(sorgu, esik):
    if not supabase: return []
    try:
        res = supabase.table("kararlar").select("*").execute()
        data = res.data
    except: return []
    if not data: return []

    sorgu_v = model.encode(sorgu, convert_to_tensor=False).astype(np.float32)
    sorgu_lower = sorgu.lower()
    sonuclar = []

    for row in data:
        try:
            db_v = np.array(json.loads(row['vektor'])).astype(np.float32)
            skor = util.cos_sim(sorgu_v, db_v).item()
            bonus = 0.30 if sorgu_lower in row['metin'].lower() else 0.0
            total = skor + bonus
            if total >= esik:
                sonuclar.append({'metin': row['metin'], 'skor': total, 'bonus': bonus})
        except: continue
    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 5. ARAYÜZ SEKMELERİ ---

st.title("⚖️ Yargıtay AI & OCR Sistemi")

tab1, tab2 = st.tabs(["📤 Yükleme", "🔍 Arama"])

with tab1:
    files = st.file_uploader("Dosya Seç (Çoklu)", accept_multiple_files=True)
    if files and st.button("Kaydet", type="primary"):
        bar = st.progress(0)
        basarili = 0
        for i, f in enumerate(files):
            try:
                img = Image.open(f)
                txt = ocr_isleme(img)
                if len(txt) > 10:
                    vec = model.encode(txt, convert_to_tensor=False).astype(np.float32)
                    if not mukerrer_kontrol(vec):
                        if veritabanina_kaydet(txt, vec): basarili += 1
            except: pass
            bar.progress((i+1)/len(files))
        st.success(f"İşlem bitti. {basarili} yeni karar eklendi.")

with tab2:
    c1, c2 = st.columns([3,1])
    with c1: q = st.text_input("Arama")
    with c2: tr = st.slider("Hassasiyet", 0.0, 1.0, 0.25)
    
    if st.button("Ara"):
        if not q: st.warning("Kelime girin")
        else:
            with st.spinner("Aranıyor..."):
                res = arama_yap_hibrit(q, tr)
                if res:
                    for r in res:
                        st.info(f"Puan: %{int(r['skor']*100)} - {r['metin'][:300]}...")
                else:
                    st.warning("Sonuç yok")
