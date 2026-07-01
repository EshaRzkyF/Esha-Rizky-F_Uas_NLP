import streamlit as st
import json
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize


st.set_page_config(
    page_title="IndoSum Summarizer - TF-IDF",
    page_icon="📄",
    layout="wide"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset", "indosum")

def load_jsonl_files(directory, prefix):
    data = []
    for filename in sorted(os.listdir(directory)):
        if filename.startswith(prefix) and filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
    return data

def flatten_doc(paragraphs):
    sentences = []
    for para in paragraphs:
        for sent in para:
            sentences.append(" ".join(sent))
    return " ".join(sentences)

def flatten_summary(summary):
    return " ".join([" ".join(sent) for sent in summary])

@st.cache_resource
def load_data():
    test = load_jsonl_files(DATA_DIR, "test")
    docs_test = [flatten_doc(d["paragraphs"]) for d in test]
    return docs_test, test

def clean_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-zA-Z0-9., ]", "", text)
    return text.lower()

def preprocess(doc):
    doc = clean_text(doc)
    try:
        return sent_tokenize(doc)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        return sent_tokenize(doc)

def summarize(document, n=3):
    sentences = preprocess(document)
    if len(sentences) <= n:
        return " ".join(sentences)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores = np.sum(tfidf_matrix.toarray(), axis=1)
    ranked = np.argsort(scores)[::-1]
    selected = sorted(ranked[:n])
    return " ".join([sentences[i] for i in selected])

st.sidebar.title("📄 IndoSum Summarizer")
st.sidebar.markdown("**TF-IDF Extractive Summarization**")
st.sidebar.markdown("---")
st.sidebar.markdown("**Info**")
st.sidebar.markdown("- Dataset: IndoSum (~71K artikel)")
st.sidebar.markdown("- Metode: TF-IDF + skoring kalimat")
st.sidebar.markdown("- Top-3 kalimat sebagai ringkasan")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dibuat oleh**")
st.sidebar.markdown("Esha Rizky Filliansyah — 230111004")
st.sidebar.markdown("NLP — Semester 6, Juni 2026")

st.title("📄 Peringkasan Teks Otomatis Bahasa Indonesia")
st.markdown("Model *extractive summarization* menggunakan **TF-IDF** untuk memilih kalimat terpenting dari artikel berita.")

with st.expander("ℹ️ **Cara Kerja TF-IDF Summarization**"):
    st.markdown("""
    **TF-IDF (Term Frequency — Inverse Document Frequency)** adalah teknik pembobotan kata:
    - **TF** = seberapa sering suatu kata muncul dalam satu kalimat
    - **IDF** = seberapa jarang kata itu muncul di seluruh kalimat (makin jarang → makin informatif)
    - **Skor kalimat** = total seluruh nilai TF-IDF term dalam kalimat tersebut
    - Kalimat dengan skor tertinggi dipilih sebagai ringkasan

    *Pipeline:* Teks → Cleaning → Tokenisasi Kalimat → TF-IDF Vectorizer → Skoring → Rank → Top-n → Ringkasan
    """)

try:
    docs_test, test_data = load_data()
except Exception as e:
    st.error(f"Gagal load dataset: {e}")
    st.info("Pastikan folder `dataset/indosum/` berisi file `test.*.jsonl`.")
    st.stop()

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Teks")
    input_mode = st.radio("Pilih mode input:", ["Acak dari dataset", "Tulis sendiri"], horizontal=True)

    if "sample_idx" not in st.session_state:
        st.session_state.sample_idx = 0

    if input_mode == "Acak dari dataset":
        if st.button("🎲 Ambil sampel acak"):
            st.session_state.sample_idx = np.random.randint(0, len(docs_test))
        idx = st.session_state.sample_idx
        input_text = docs_test[idx]
        sample_cat = test_data[idx]["category"] if idx < len(test_data) else "-"
        st.info(f"Sample #{idx} — {len(input_text)} karakter | Kategori: **{sample_cat}**")
        with st.expander("Lihat teks lengkap"):
            st.write(input_text)
    else:
        input_text = st.text_area("Masukkan teks berita:", height=250,
            placeholder="Tempel teks berita Indonesia di sini...")

    n_sentences = st.slider("Jumlah kalimat ringkasan:", 1, 10, 3,
        help="Makin besar n, makin panjang ringkasannya. n=3 memberikan keseimbangan optimal.")

    st.caption("💡 **Tips:** Untuk artikel berita umum, n=3 sudah cukup. "
               "Gunakan n>3 untuk dokumen panjang atau jika ingin ringkasan lebih detail.")

with col2:
    st.subheader("Hasil Ringkasan")
    if st.button("✂️ Ringkas Sekarang", type="primary") or input_mode == "Acak dari dataset":
        with st.spinner("Merangkas teks..."):
            result = summarize(input_text, n_sentences)
        st.success("Ringkasan berhasil dibuat!")
        st.write(result)

        with st.expander("📊 Lihat Skor & Peringkat Semua Kalimat"):
            st.caption("Kalimat dengan skor TF-IDF tertinggi akan dipilih sebagai ringkasan. "
                      "Semakin tinggi skor → semakin representatif kalimat tersebut.")
            sentences = preprocess(input_text)
            if len(sentences) > n_sentences:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(sentences)
                scores = np.sum(tfidf_matrix.toarray(), axis=1)
                ranked = np.argsort(scores)[::-1]
                selected = sorted(ranked[:n_sentences])
                for i, sent in enumerate(sentences):
                    prefix = "✅ **TERPILIH**" if i in selected else "   "
                    marker = "★" if i in selected else " "
                    st.markdown(f"{marker} **[{i+1}]** Skor: `{scores[i]:.4f}`  {prefix}")
                    st.write(sentences[i][:150] + ("..." if len(sentences[i]) > 150 else ""))
                    if i < len(sentences) - 1:
                        st.markdown("---")
            else:
                st.info("Jumlah kalimat ≤ n, semua kalimat digunakan.")
    else:
        st.info("Klik tombol **Ringkas Sekarang** atau pilih **Acak dari dataset**.")
