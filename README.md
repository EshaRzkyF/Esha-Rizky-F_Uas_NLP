# Peringkasan Teks Otomatis Bahasa Indonesia

**TF-IDF Extractive Summarization** pada dataset **IndoSum**

Proyek ini adalah implementasi *extractive text summarization* untuk artikel berita bahasa Indonesia menggunakan metode pembobotan **TF-IDF**. Model memilih top-n kalimat dengan skor TF-IDF tertinggi sebagai ringkasan.

---

## 🎯 Latar Belakang & Motivasi

### Masalah
- **Ledakan informasi digital** → pembaca kesulitan membaca artikel lengkap
- **Kebutuhan ringkasan cepat** → untuk berita dan dokumen panjang
- **Minimnya eksplorasi** metode sederhana (non-deep learning) pada dataset bahasa Indonesia

### Motivasi
- **TF-IDF**: metode klasik, cepat, transparan, tanpa GPU — cocok untuk deployment terbatas
- **IndoSum**: benchmark dataset resmi untuk text summarization bahasa Indonesia
- **Baseline solid**: membandingkan unsupervised method vs deep learning approaches

### Pertanyaan Penelitian
> "Seberapa baik TF-IDF extractive summarization dalam meringkas artikel berita bahasa Indonesia dibandingkan dengan ringkasan manual?"

---

## 📂 Struktur Proyek

```
├── notebook/
│   ├── NLP_UAS.ipynb          # Notebook utama (pipeline lengkap)
│   └── model_summarizer.pkl   # Model yang sudah disimpan (pickle)
├── dataset/
│   └── indosum/                # Dataset IndoSum (15 file JSONL)
├── app.py                      # Aplikasi demo Streamlit
├── ppt.md                      # Slide presentasi
└── README.md
```

## 📊 Dataset

**IndoSum** (Kurniawan & Louvan, 2018) — *benchmark dataset* untuk peringkasan teks bahasa Indonesia.

| Item              | Detail                          |
|-------------------|---------------------------------|
| Total artikel     | 71.353                          |
| Train (80%)       | 57.082                          |
| Test (20%)        | 14.271                          |
| Kategori          | 6 (olahraga, hiburan, showbiz, teknologi, inspirasi, tajuk utama) |
| Format            | JSONL (paragraphs, summary, gold_labels, category) |

## ⚙️ Pipeline

```
Input Teks → Cleaning → Tokenisasi Kalimat → TF-IDF Vectorizer → Skoring → Rank → Top-n → Ringkasan
```

### Metode

**TF-IDF (Term Frequency — Inverse Document Frequency)**:
- **TF** = frekuensi kata dalam suatu kalimat
- **IDF** = log(total kalimat / kalimat yang mengandung kata tersebut)
- **Skor kalimat** = jumlah seluruh nilai TF-IDF term dalam kalimat
- **n kalimat** dengan skor tertinggi dipilih sebagai ringkasan

## 📈 Hasil Evaluasi

### ROUGE Score (100 sampel, n=3)

| Metrik    | Precision | Recall | F1     |
|-----------|-----------|--------|--------|
| ROUGE-1   | 0.3811    | 0.5214 | 0.4355 |
| ROUGE-2   | 0.2728    | 0.3687 | 0.3103 |
| ROUGE-L   | 0.3205    | 0.4334 | 0.3646 |

### Confusion Matrix — Sentence Selection (200 dokumen)

| Metrik     | Nilai  |
|------------|--------|
| Accuracy   | 0.7470 |
| Precision  | 0.3300 |
| Recall     | 0.2849 |
| F1-Score   | 0.3058 |

### ROUGE-1 per Kategori

| Kategori    | ROUGE-1 |
|-------------|---------|
| Olahraga    | 0.4988  |
| Tajuk Utama | 0.4402  |
| Hiburan     | 0.4345  |
| Inspirasi   | 0.3859  |
| Teknologi   | 0.3700  |
| Showbiz     | 0.3508  |

## � Analisis & Insight

### Kinerja Model
- **ROUGE-1 F1 (0.4355)** menunjukkan model cukup baik dalam menangkap unigram dari summary referensi
- **Recall (0.5214) > Precision (0.3811)** → model cenderung memilih terlalu banyak kalimat, trade-off untuk mencakup informasi penting
- **Variasi score per kategori (0.35–0.50)** menunjukkan performa bergantung pada tipe artikel dan struktur konten
- **Accuracy 74.7%** pada sentence selection indicates reasonable performance pada binary decision task

### Performa Terbaik & Terburuk
- **Terbaik**: Olahraga (ROUGE-1: 0.4988)
  - Struktur artikel lebih teratur (lead paragraph + detail)
  - Istilah olahraga lebih konsisten antar artikel
  
- **Terburuk**: Showbiz (ROUGE-1: 0.3508)
  - Konten lebih subjektif dan narrative-driven
  - Informasi penting tersebar di berbagai bagian artikel

## ⚡ Limitasi

1. **Metode Unsupervised**
   - TF-IDF tidak mempelajari pola bahasa spesifik → generalisasi terbatas
   - Tidak bisa mengidentifikasi kalimat kontekstual yang penting

2. **Extractive-Only Approach**
   - Hanya memilih kalimat existing, tidak bisa parafrase atau membuat summary baru
   - Ringkasan bisa terasa "patah-patah" tanpa transisi alami

3. **Sentence-Level Processing**
   - Tidak mempertimbangkan hubungan semantic antar kalimat
   - Kehilangan koherensi global dokumen

4. **Fixed Parameter (n=3)**
   - Tidak adaptif terhadap panjang dokumen
   - Compression ratio tetap ~33% untuk semua artikel

5. **Cleaning yang Sederhana**
   - Menghapus karakter khusus bisa hilangkan informasi penting (nama, angka)
   - Belum menggunakan stemming/lemmatization Sastrawi optimal

## 🔮 Future Work

1. **Metode Alternatif**
   - Implementasi TextRank / LexRank untuk comparison
   - Eksperimen dengan abstractive summarization (BERT, T5)

2. **Parameter Tuning**
   - Adaptive n berdasarkan document length
   - Eksperimen dengan TF-IDF variants (sublinear TF, L2 normalization)
   - Weighted sentence scoring (bonus untuk lead paragraph)

3. **Preprocessing Improvement**
   - Integrasikan Sastrawi stemming untuk normalized tokens
   - Named Entity Recognition (NER) untuk preserve informasi penting
   - Stopword removal yang lebih sophisticated

4. **Evaluasi Lebih Mendalam**
   - Human evaluation (kualitas konten ringkasan)
   - Correlation ROUGE metrics dengan readability scores
   - Cross-category analysis untuk pattern discovery

## �🚀 Cara Menjalankan

### 1. Clone repositori

```bash
git clone https://github.com/esharizky/indosum-summarizer.git
cd indosum-summarizer
```

### 2. Install dependensi

```bash
pip install nltk scikit-learn rouge-score Sastrawi seaborn streamlit
```

### 3. Jalankan notebook (eksperimen & evaluasi)

```bash
jupyter notebook notebook/NLP_UAS.ipynb
```

### 4. Jalankan aplikasi Streamlit (demo)

```bash
streamlit run app.py
```

## 🛠️ Library yang Digunakan

- **Python 3.11**
- **scikit-learn** — TfidfVectorizer
- **NLTK** — tokenisasi kalimat
- **rouge-score** — evaluasi ROUGE
- **Streamlit** — aplikasi demo
- **matplotlib / seaborn** — visualisasi
- **numpy / pandas** — manipulasi data

## 📌 Catatan

- Model menyimpan fungsi `summarize()` ke `model_summarizer.pkl` menggunakan pickle
- Dataset IndoSum tidak termasuk dalam repositori ini (unduh dari [sumber resmi](https://github.com/kawatkawat/IndoSum))
- n=3 memberikan keseimbangan optimal antara informasi dan panjang ringkasan

## � Referensi

1. **Kurniawan, K., & Louvan, S.** (2018). "IndoSum: A new benchmark dataset for Indonesian text summarization." *Proceedings of PACLIC 32*, 224–232.

2. **Lin, C. Y.** (2004). "ROUGE: A package for automatic evaluation of summaries." *In Proceedings of ACL Workshop on Text Summarization Branches Out*.

3. **Sparck Jones, K.** (1972). "A statistical interpretation of term specificity and its application in retrieval." *Journal of Documentation*, 28(1), 11–21. — Original TF-IDF paper

4. **Mihalcea, R., & Tarau, P.** (2004). "TextRank: Bringing order into texts." *Proceedings of EMNLP*, 404–411. — Alternative extractive method

5. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K.** (2018). "BERT: Pre-training of deep bidirectional transformers for language understanding." *arXiv preprint arXiv:1810.04805*. — Modern abstractive approach

## �👤 Identitas

**Esha Rizky Filliansyah — 230111004**  
Natural Language Processing — Semester 6, Juni 2026
