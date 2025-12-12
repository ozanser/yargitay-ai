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

# --- 2. GÜVENLİK ---
if 'giris_yapildi' not in st.session_state:
    st.session_state['giris_yapildi'] = False

if not st.session_state['giris_yapildi']:
    st.markdown("## 🔒 Güvenli Yargıtay Sistemi")
    gecerli_sifre = "1234"
    try:
        if "APP_PASSWORD" in st.secrets: gecerli_sifre = st.secrets["APP_PASSWORD"]
    except: pass

    girilen = st.text_input("Şifre:", type="password")
    if st.button("Giriş Yap", type="primary"):
        if girilen == gecerli_sifre:
            st.session_state['giris_yapildi'] = True
            st.rerun()
        else: st.error("Hatalı Şifre")
    st.stop()

# --- 3. BAĞLANTILAR ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    SUPABASE_URL = "URL"
    SUPABASE_KEY = "KEY"

@st.cache_resource
def init_supabase():
    try: return create_client(SUPABASE_URL, SUPABASE_KEY)
    except: return None

supabase = init_supabase()

@st.cache_resource
def model_yukle():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = model_yukle()

# --- 4. YARDIMCI FONKSİYONLAR (YENİ TÜRKÇE YAMASI) ---

def turkce_kucult(text):
    """
    Python'un standart lower() fonksiyonu Türkçe 'İ' ve 'I' harflerini bozar.
    Bu fonksiyon onları doğru şekilde 'i' ve 'ı' yapar.
    """
    if not text: return ""
    text = text.replace("İ", "i").replace("I", "ı").replace("Ğ", "ğ").replace("Ü", "ü").replace("Ş", "ş").replace("Ö", "ö").replace("Ç", "ç")
    return text.lower()

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
            if util.cos_sim(yeni_v, db_v).item() > 0.95: return True
        except: continue
    return False

def akilli_temizlik():
    if not supabase: return 0
    res = supabase.table("kararlar").select("id, vektor").execute()
    if not res.data: return 0
    silinecek, saklanan = [], []
    for satir in res.data:
        try:
            v_curr = np.array(json.loads(satir['vektor'])).astype(np.float32)
            kopya = False
            for _, v_sakli in saklanan:
                if util.cos_sim(v_curr, v_sakli).item() > 0.95:
                    kopya = True
                    break
            if kopya: silinecek.append(satir['id'])
            else: saklanan.append((satir['id'], v_curr))
        except: continue
    if silinecek:
        for i in range(0, len(silinecek), 20):
            supabase.table("kararlar").delete().in_("id", silinecek[i:i+20]).execute()
    return len(silinecek)

def veritabani_sifirla():
    if not supabase: return False
    res = supabase.table("kararlar").select("id").execute()
    ids = [d['id'] for d in res.data]
    if ids:
        for i in range(0, len(ids), 20):
             supabase.table("kararlar").delete().in_("id", ids[i:i+20]).execute()
    return True

# --- 5. ARAMA MOTORU (GÜÇLENDİRİLMİŞ) ---
def arama_yap_gorsel(sorgu):
    if not supabase: return []
    try: res = supabase.table("kararlar").select("*").execute()
    except: return []
    if not res.data: return []

    sorgu_v = model.encode(sorgu, convert_to_tensor=False).astype(np.float32)
    
    # Türkçe karakterlere uygun küçültme yapıyoruz
    sorgu_kucuk = turkce_kucult(sorgu)
    
    sonuclar = []

    for row in res.data:
        try:
            db_v = np.array(json.loads(row['vektor'])).astype(np.float32)
            skor = util.cos_sim(sorgu_v, db_v).item()
            
            # KELİME BONUSU (ARTIRILDI VE DÜZELTİLDİ)
            bonus = 0.0
            db_metin_kucuk = turkce_kucult(row['metin'])
            
            if sorgu_kucuk in db_metin_kucuk:
                # Eğer kelime geçiyorsa puanı direkt +0.50 artır!
                bonus = 0.50 
            
            toplam = skor + bonus
            
            # Eğer bonusla 1.0'ı geçerse 0.99'a sabitle (Estetik için)
            if toplam > 0.99: toplam = 0.99
            
            if toplam > 0.25:
                sonuclar.append({'metin': row['metin'], 'skor': toplam})
        except: continue
    
    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 6. ARAYÜZ ---
st.title("⚖️ Yargıtay AI & OCR Sistemi")

# Yan Menü
with st.sidebar:
    st.success("✅ Yetkili Girişi")
    if st.button("🚪 Çıkış"):
        st.session_state['giris_yapildi'] = False
        st.rerun()
    st.divider()
    st.header("⚙️ Yönetim")
    if supabase:
        try:
            sayi = supabase.table("kararlar").select("id", count="exact").execute().count
            st.metric("Toplam Karar", sayi)
            st.write("---")
            if st.button("🧹 Akıllı Temizlik", type="primary"):
                with st.spinner("Taranıyor..."):
                    s = akilli_temizlik()
                    if s: st.success(f"{s} silindi"); time.sleep(1); st.rerun()
                    else: st.info("Temiz")
            with st.expander("⚠️ Tehlikeli"):
                if st.button("TÜMÜNÜ SİL"): veritabani_sifirla(); st.warning("Silindi!"); time.sleep(1); st.rerun()
        except: st.error("Bağlantı Yok")

tab1, tab2 = st.tabs(["📤 Yükleme", "🔍 Arama"])

with tab1:
    files = st.file_uploader("Dosya Seç", accept_multiple_files=True)
    if files and st.button("Kaydet", type="primary"):
        bar = st.progress(0)
        basarili = 0
        mukerrer_sayi = 0
        for i, f in enumerate(files):
            try:
                img = Image.open(f)
                txt = ocr_isleme(img)
                if len(txt) > 10:
                    v = model.encode(txt, convert_to_tensor=False).astype(np.float32)
                    if mukerrer_kontrol(v): mukerrer_sayi += 1
                    else:
                        if veritabanina_kaydet(txt, v): basarili += 1
            except: pass
            bar.progress((i+1)/len(files))
        st.success("Bitti")
        c1, c2 = st.columns(2)
        c1.metric("Eklenen", basarili)
        c2.metric("Mükerrer", mukerrer_sayi)

with tab2:
    sorgu = st.text_input("Arama:", placeholder="Örn: eroin")
    if st.button("Ara"):
        if sorgu:
            with st.spinner("Aranıyor..."):
                res = arama_yap_gorsel(sorgu)
                if res:
                    st.success(f"{len(res)} sonuç.")
                    for r in res:
                        st.markdown("---")
                        c1, c2 = st.columns([1, 4])
                        with c1:
                            puan = int(r['skor'] * 100)
                            st.metric("Uygunluk", f"%{puan}")
                            if puan > 80: st.success("Yüksek")
                            elif puan > 50: st.warning("Orta")
                            else: st.info("Düşük")
                        with c2: st.info(r['metin'])
                else: st.warning("Sonuç yok")
        else: st.warning("Yazınız.")
