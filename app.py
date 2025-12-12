import streamlit as st
from PIL import Image, ImageEnhance
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import json
import time

# --- 1. AYARLAR ---
st.set_page_config(page_title="Yargıtay AI (Tamir)", layout="wide", page_icon="⚖️")

# --- 2. GÜVENLİK ---
if 'giris_yapildi' not in st.session_state:
    st.session_state['giris_yapildi'] = False

if not st.session_state['giris_yapildi']:
    st.markdown("## 🔒 Güvenli Yargıtay Sistemi")
    gecerli_sifre = "1234"
    try:
        if "APP_PASSWORD" in st.secrets:
            gecerli_sifre = st.secrets["APP_PASSWORD"]
    except: pass

    girilen = st.text_input("Şifre:", type="password")
    if st.button("Giriş Yap", type="primary"):
        if girilen == gecerli_sifre:
            st.session_state['giris_yapildi'] = True
            st.rerun()
        else:
            st.error("Hatalı Şifre")
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

# --- 4. GELİŞMİŞ TEMİZLİK ROBOTU (YENİ) ---
def akilli_temizlik():
    """
    Metin eşleşmesine değil, VEKTÖR (Anlam) eşleşmesine bakar.
    OCR hatalarına rağmen kopyaları bulur.
    """
    if not supabase: return 0
    
    # Tüm vektörleri çek
    res = supabase.table("kararlar").select("id, vektor").execute()
    data = res.data
    if not data: return 0

    silinecek_idler = []
    saklanan_vektorler = [] # (id, numpy_vektor)

    for satir in data:
        try:
            # Mevcut satırın vektörü
            su_anki_vektor = np.array(json.loads(satir['vektor'])).astype(np.float32)
            
            kopya_mi = False
            # Daha önce sakladıklarımızla karşılaştır
            for sakli_id, sakli_vektor in saklanan_vektorler:
                skor = util.cos_sim(su_anki_vektor, sakli_vektor).item()
                
                # EŞİK: %95'ten fazla benziyorsa bu bir kopyadır
                if skor > 0.95:
                    kopya_mi = True
                    break
            
            if kopya_mi:
                silinecek_idler.append(satir['id'])
            else:
                saklanan_vektorler.append((satir['id'], su_anki_vektor))
                
        except: continue

    # Toplu Silme
    if silinecek_idler:
        # Supabase API limiti olduğu için 20'şer 20'şer siliyoruz
        chunk_size = 20
        for i in range(0, len(silinecek_idler), chunk_size):
            chunk = silinecek_idler[i:i + chunk_size]
            supabase.table("kararlar").delete().in_("id", chunk).execute()
            
    return len(silinecek_idler)

def veritabani_sifirla():
    """HER ŞEYİ SİLER (Dikkatli Kullanın)"""
    if not supabase: return False
    # Tüm ID'leri çekip siler
    res = supabase.table("kararlar").select("id").execute()
    ids = [d['id'] for d in res.data]
    if ids:
        for i in range(0, len(ids), 20):
             supabase.table("kararlar").delete().in_("id", ids[i:i+20]).execute()
    return True

# --- 5. İŞLEM FONKSİYONLARI ---
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
    # Kayıt anında hızlı kontrol
    if not supabase: return False
    res = supabase.table("kararlar").select("vektor").execute()
    if not res.data: return False
    yeni_v = yeni_v.astype(np.float32)
    for row in res.data:
        try:
            db_v = np.array(json.loads(row['vektor'])).astype(np.float32)
            if util.cos_sim(yeni_v, db_v).item() > 0.95: return True # Eşik %95
        except: continue
    return False

def arama_yap_gorsel(sorgu):
    """Eski güzel yüzde göstergeli arama"""
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
            
            # Kelime Bonusu (Varsa %20 ekle ama 100'ü geçirme)
            bonus = 0.0
            if sorgu_lower in row['metin'].lower():
                bonus = 0.20
            
            toplam = skor + bonus
            
            # Baraj: %25 altı çöp
            if toplam > 0.25:
                sonuclar.append({'metin': row['metin'], 'skor': toplam})
        except: continue
    
    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 6. ARAYÜZ ---

st.title("⚖️ Yargıtay AI & OCR Sistemi")

# YAN MENÜ (YÖNETİM)
with st.sidebar:
    st.success("✅ Yetkili Girişi")
    if st.button("🚪 Çıkış"):
        st.session_state['giris_yapildi'] = False
        st.rerun()
    
    st.divider()
    st.header("⚙️ Veritabanı Yönetimi")
    
    if supabase:
        try:
            sayi = supabase.table("kararlar").select("id", count="exact").execute().count
            st.metric("Toplam Karar", sayi)
            
            st.write("---")
            st.write("🔧 Bakım Araçları")
            
            # GELİŞMİŞ TEMİZLİK BUTONU
            if st.button("🧹 Akıllı Temizlik (Vektör)", type="primary"):
                with st.spinner("Yapay Zeka kopyaları arıyor..."):
                    silinen = akilli_temizlik()
                    if silinen > 0:
                        st.success(f"{silinen} kopya silindi!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.info("Kopya bulunamadı.")
            
            st.write("")
            # SIFIRLAMA BUTONU (Kırmızı)
            with st.expander("⚠️ Tehlikeli Bölge"):
                if st.button("Tüm Veritabanını SİL"):
                    veritabani_sifirla()
                    st.warning("Veritabanı sıfırlandı!")
                    time.sleep(1)
                    st.rerun()

        except: st.error("Bağlantı Hatası")

# ANA EKRAN
tab1, tab2 = st.tabs(["📤 Karar Yükle", "🔍 Arama Yap"])

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
                    # Yüklerken de sıkı kontrol
                    if mukerrer_kontrol(v):
                        mukerrer_sayi += 1
                    else:
                        if veritabanina_kaydet(txt, v): basarili += 1
            except: pass
            bar.progress((i+1)/len(files))
        
        st.success(f"İşlem bitti.")
        c1, c2 = st.columns(2)
        c1.metric("Eklenen", basarili)
        c2.metric("Mükerrer (Atlandı)", mukerrer_sayi)

with tab2:
    sorgu = st.text_input("Arama Terimi:", placeholder="Örn: kıdem tazminatı faiz")
    if st.button("Ara"):
        if sorgu:
            with st.spinner("Taranıyor..."):
                res = arama_yap_gorsel(sorgu)
                if res:
                    st.success(f"{len(res)} sonuç bulundu.")
                    for r in res:
                        st.markdown("---")
                        # ESKİ GÜZEL GÖRÜNÜM GERİ GELDİ
                        c1, c2 = st.columns([1, 4])
                        with c1:
                            # Büyük Puan Göstergesi
                            puan = int(r['skor'] * 100)
                            if puan > 100: puan = 100 # Bonusla 100'ü geçerse düzelt
                            
                            st.metric("Uygunluk", f"%{puan}")
                            
                            if puan > 80:
                                st.success("Çok Yüksek")
                            elif puan > 50:
                                st.warning("Orta")
                            else:
                                st.info("Düşük")
                                
                        with c2:
                            st.info(r['metin'])
                else:
                    st.warning("Sonuç bulunamadı.")
        else:
            st.warning("Lütfen bir şey yazın.")
