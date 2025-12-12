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

# --- 4. TEMİZLİK VE YÖNETİM ---
def akilli_temizlik():
    if not supabase: return 0
    res = supabase.table("kararlar").select("id, vektor").execute()
    if not res.data: return 0

    silinecek = []
    saklanan = [] # (id, vektor)

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
        # Parça parça sil (Timeout yememek için)
        chunk = 20
        for i in range(0, len(silinecek), chunk):
            supabase.table("kararlar").delete().in_("id", silinecek[i:i+chunk]).execute()
    return len(silinecek)

def veritabani_sifirla():
    if not supabase: return False
    res = supabase.table("kararlar").select("id").execute()
    ids = [d['id'] for d in res.data]
    if ids:
        for i in range(0, len(ids), 20):
             supabase.table("kararlar").delete().in_("id", ids[i:i+20]).execute()
    return True

# --- 5. YAN MENÜ (DÜZELTİLDİ) ---
with st.sidebar:
    st.success("✅ Yetkili Girişi")
    if st.button("🚪 Çıkış"):
        st.session_state['giris_yapildi'] = False
        st.rerun()
    
    st.divider()
    st.header("⚙️ Veritabanı")
    
    if supabase:
        # İstatistik Gösterimi (Hata verirse sadece burası hata verir)
        try:
            sayi = supabase.table("kararlar").select("id", count="exact").execute().count
            st.metric("Toplam Karar", sayi)
        except Exception as e:
            st.caption(f"Sayaç hatası: {e}")

        st.write("---")
        
        # TEMİZLİK BUTONU (Try-Except Bloğundan Çıkarıldı)
        if st.button("🧹 Akıllı Temizlik", type="primary"):
            with st.spinner("Kopyalar aranıyor..."):
                try:
                    silinen = akilli_temizlik()
                    if silinen > 0:
                        st.success(f"{silinen} kopya silindi!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.info("Kopya yok.")
                except Exception as e:
                    st.error(f"Temizlik hatası: {e}")

        # SIFIRLAMA BUTONU
        with st.expander("⚠️ Tehlikeli Bölge"):
            if st.button("TÜMÜNÜ SİL"):
                try:
                    veritabani_sifirla()
                    st.warning("Veritabanı sıfırlandı!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Silme hatası: {e}")
    else:
        st.error("Supabase Bağlantısı Yok")

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
            if util.cos_sim(yeni_v, db_v).item() > 0.95: return True
        except: continue
    return False

def arama_yap_gorsel(sorgu):
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
            bonus = 0.20 if sorgu_lower in row['metin'].lower() else 0.0
            toplam = skor + bonus
            if toplam > 0.25:
                sonuclar.append({'metin': row['metin'], 'skor': toplam})
        except: continue
    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

# --- 7. ARAYÜZ ---
st.title("⚖️ Yargıtay AI & OCR Sistemi")

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
                    if mukerrer_kontrol(v):
                        mukerrer_sayi += 1
                    else:
                        if veritabanina_kaydet(txt, v): basarili += 1
            except: pass
            bar.progress((i+1)/len(files))
        st.success("İşlem Tamamlandı")
        c1, c2 = st.columns(2)
        c1.metric("Eklenen", basarili)
        c2.metric("Mükerrer (Atlandı)", mukerrer_sayi)

with tab2:
    sorgu = st.text_input("Arama:", placeholder="Örn: kıdem tazminatı")
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
                            if puan > 100: puan = 100
                            st.metric("Uygunluk", f"%{puan}")
                            if puan > 80: st.success("Yüksek")
                            elif puan > 50: st.warning("Orta")
                            else: st.info("Düşük")
                        with c2: st.info(r['metin'])
                else: st.warning("Sonuç yok")
