import streamlit as st
from PIL import Image, ImageEnhance
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import json
import time

# --- 1. AYARLAR ---
st.set_page_config(
    page_title="İçtihat Ekleme ve Arama", 
    layout="wide", 
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# --- 2. GÜVENLİK VE GİRİŞ ---
if 'giris_yapildi' not in st.session_state:
    st.session_state['giris_yapildi'] = False

if not st.session_state['giris_yapildi']:
    # TASARIM DÜZELTME: color: #333 eklendi (Yazıları siyah yapar)
    st.markdown("""
    <style>
    .login-container {
        padding: 40px;
        border-radius: 12px;
        background-color: #ffffff;
        color: #333333; /* YAZI RENGİ SİYAH OLARAK ZORLANDI */
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 50px;
        border-top: 6px solid #d32f2f;
    }
    /* Kutunun içindeki başlıkları da siyah yap */
    .login-container h1, .login-container h3 {
        color: #333333 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # "Yetkili Personel Girişi" yazısı kaldırıldı
        st.markdown("""
        <div class='login-container'>
            <h1 style='font-size: 3rem; margin-bottom: 0;'>⚖️</h1>
            <h3 style='font-weight: 600; margin-top: 10px;'>Yargıtay İçtihat Ekleme ve Arama</h3>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        
        with st.form("giris_formu"):
            sifre = st.text_input("Erişim Şifresi", type="password")
            submit_btn = st.form_submit_button("Sisteme Giriş Yap", type="primary", use_container_width=True)
            
            if submit_btn:
                gercek_sifre = "1234"
                try:
                    if "APP_PASSWORD" in st.secrets:
                        gercek_sifre = st.secrets["APP_PASSWORD"]
                except: pass

                if sifre == gercek_sifre:
                    st.session_state['giris_yapildi'] = True
                    st.success("Giriş Başarılı!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Hatalı Şifre!")
    st.stop()

# ====================================================
# İÇERİK (GİRİŞ YAPANLAR İÇİN)
# ====================================================

# --- 3. TASARIM (CSS) ---
st.markdown("""
<style>
.decision-card {
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    border-left: 5px solid #d32f2f;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.badge-high { background-color: #2e7d32; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }
.badge-med { background-color: #f9a825; color: black; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }
.badge-low { background-color: #c62828; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }
.bonus-tag { background-color: #e3f2fd; color: #1565c0; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold; margin-left: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 4. BAĞLANTILAR ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    SUPABASE_URL = ""
    SUPABASE_KEY = ""

@st.cache_resource
def init_supabase():
    try: return create_client(SUPABASE_URL, SUPABASE_KEY)
    except: return None

supabase = init_supabase()

@st.cache_resource
def model_yukle():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = model_yukle()

# --- 5. FONKSİYONLAR ---
def turkce_kucult(text):
    if not text: return ""
    return text.replace("İ", "i").replace("I", "ı").lower()

def ocr_isleme(image):
    img = image.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    try: return pytesseract.image_to_string(img, lang='tur')
    except: return pytesseract.image_to_string(img)

def veritabanina_kaydet(metin, vektor):
    if not supabase: return False
    try:
        data = {"metin": metin, "vektor": json.dumps(vektor.tolist())}
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
    for row in res.data:
        try:
            curr = np.array(json.loads(row['vektor'])).astype(np.float32)
            kopya = False
            for _, sakli in saklanan:
                if util.cos_sim(curr, sakli).item() > 0.95:
                    kopya = True; break
            if kopya: silinecek.append(row['id'])
            else: saklanan.append((row['id'], curr))
        except: continue
    if silinecek:
        for i in range(0, len(silinecek), 20):
            supabase.table("kararlar").delete().in_("id", silinecek[i:i+20]).execute()
    return len(silinecek)

def veritabani_sifirla():
    if not supabase: return
    res = supabase.table("kararlar").select("id").execute()
    ids = [d['id'] for d in res.data]
    for i in range(0, len(ids), 20):
        supabase.table("kararlar").delete().in_("id", ids[i:i+20]).execute()

def arama_yap_gorsel(sorgu, esik):
    if not supabase: return []
    try: res = supabase.table("kararlar").select("*").execute()
    except: return []
    if not res.data: return []

    sorgu_v = model.encode(sorgu, convert_to_tensor=False).astype(np.float32)
    sorgu_kucuk = turkce_kucult(sorgu)
    sonuclar = []

    for row in res.data:
        try:
            db_v = np.array(json.loads(row['vektor'])).astype(np.float32)
            skor = util.cos_sim(sorgu_v, db_v).item()
            bonus = 0.50 if sorgu_kucuk in turkce_kucult(row['metin']) else 0.0
            total = skor + bonus
            if total > 0.99: total = 0.99
            
            if total >= esik:
                sonuclar.append({'metin': row['metin'], 'skor': total, 'bonus': bonus})
        except: continue
    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 6. ARAYÜZ ---

# YAN MENÜ TASARIMI
with st.sidebar:
    st.header("⚙️ Yönetim Paneli")
    
    if supabase:
        try:
            c = supabase.table("kararlar").select("id", count="exact").execute().count
            st.info(f"📚 Arşivde **{c}** karar var.")
        except: st.error("Bağlantı Yok")
    
    st.markdown("---")
    
    # Yönetim Araçları
    st.write("🔧 Araçlar")
    if st.button("🧹 Kopyaları Sil", use_container_width=True):
        n = akilli_temizlik()
        if n: st.success(f"{n} silindi"); time.sleep(1); st.rerun()
        else: st.info("Temiz")

    with st.expander("🚨 Kırmızı Alan"):
        if st.button("Her Şeyi SİL", type="primary", use_container_width=True):
            veritabani_sifirla()
            st.warning("Sıfırlandı")
            time.sleep(1); st.rerun()

    # ÇIKIŞ BUTONU EN ALTA ALINDI VE BOŞLUK EKLENDİ
    st.markdown("<br>" * 5, unsafe_allow_html=True) 
    st.markdown("---")
    if st.button("🚪 Güvenli Çıkış", type="secondary", use_container_width=True):
        st.session_state['giris_yapildi'] = False
        st.rerun()

# ANA SAYFA BAŞLIĞI
st.markdown("""
<div style="background-color:#d32f2f;padding:20px;border-radius:10px;margin-bottom:25px;">
    <h1 style="color:white;text-align:center;margin:0;">İçtihat Ekleme ve Arama Platformu</h1>
    <p style="color:#ffcdd2;text-align:center;margin-top:5px;">Yargıtay Kararları Yapay Zeka Arşivi</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📤 **Karar Yükleme Merkezi**", "🔍 **Akıllı Arama Motoru**"])

with tab1:
    st.markdown("### 📄 Dosya Yükleme")
    st.caption("Yargıtay kararlarının fotoğraflarını buraya sürükleyin. Sistem otomatik okur ve arşivler.")
    
    files = st.file_uploader("", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if files:
        if st.button(f"🚀 {len(files)} Adet Kararı İşle ve Kaydet", type="primary", use_container_width=True):
            bar = st.progress(0)
            basarili, mukerrer = 0, 0
            
            for i, f in enumerate(files):
                try:
                    img = Image.open(f)
                    txt = ocr_isleme(img)
                    if len(txt) > 10:
                        v = model.encode(txt, convert_to_tensor=False).astype(np.float32)
                        if mukerrer_kontrol(v): mukerrer += 1
                        else:
                            if veritabanina_kaydet(txt, v): basarili += 1
                except: pass
                bar.progress((i+1)/len(files))
            
            st.balloons()
            st.success(f"İşlem Tamamlandı! ✅ {basarili} Eklendi, ⛔ {mukerrer} Mükerrer.")

with tab2:
    col_s, col_f = st.columns([3, 1])
    with col_s: q = st.text_input("Arama Kelimesi", placeholder="Örn: kıdem tazminatı, uyuşturucu ticareti...", label_visibility="collapsed")
    with col_f: sens = st.slider("Hassasiyet Ayarı", 0.0, 1.0, 0.25)

    if st.button("🔎 İçtihatlarda Ara", type="primary", use_container_width=True):
        if q:
            with st.spinner("Arşiv taranıyor..."):
                res = arama_yap_gorsel(q, sens)
                if res:
                    st.markdown(f"### 🎯 {len(res)} Sonuç Bulundu")
                    for r in res:
                        p = int(r['skor']*100)
                        if p >= 80: css_class = "badge-high"; label = "Yüksek"
                        elif p >= 50: css_class = "badge-med"; label = "Orta"
                        else: css_class = "badge-low"; label = "Düşük"
                        
                        bonus_html = '<span class="bonus-tag">✅ Kelime Var</span>' if r['bonus'] > 0 else ''
                        
                        st.markdown(f"""
                        <div class="decision-card">
                            <div style="margin-bottom:8px;">
                                <span class="{css_class}">%{p} - {label}</span>
                                {bonus_html}
                            </div>
                            <div style="color:#333;">{r['metin']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else: st.warning("Sonuç bulunamadı.")
        else: st.warning("Lütfen arama kelimesi girin.")
