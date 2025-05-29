# Laporan Proyek Machine Learning - Sandy Sanjaya

## Project Overview

Dalam era digital saat ini, sistem rekomendasi telah menjadi elemen penting dalam berbagai layanan berbasis teknologi. Platform seperti YouTube, Amazon, dan Netflix telah membuktikan bagaimana algoritma rekomendasi mampu meningkatkan pengalaman pengguna secara signifikan dengan menyarankan konten yang relevan berdasarkan preferensi dan perilaku mereka. Dalam konteks ini, sistem rekomendasi tidak hanya berfungsi sebagai alat bantu, tetapi juga sebagai strategi bisnis yang dapat meningkatkan keterlibatan pengguna dan pendapatan perusahaan [1].

Secara umum, sistem rekomendasi merupakan pendekatan berbasis data yang dirancang untuk menyarankan item—seperti produk, film, atau buku—yang kemungkinan besar akan disukai oleh pengguna. Algoritma yang digunakan biasanya memanfaatkan pola interaksi historis antara pengguna dan item, baik melalui metode kolaboratif, konten, maupun pendekatan hibrida. Salah satu indikator penting dari signifikansi sistem rekomendasi adalah kompetisi “Netflix Prize” yang diselenggarakan pada tahun 2006, di mana Netflix menawarkan hadiah sebesar satu juta dolar AS kepada siapa pun yang dapat mengembangkan algoritma rekomendasi yang mengungguli sistem internal mereka [2].

Dalam ranah literasi digital dan industri buku, sistem rekomendasi memiliki potensi yang tidak kalah besar. Banyaknya jumlah buku yang tersedia membuat pembaca kesulitan menemukan buku yang sesuai dengan minat mereka. Di sinilah sistem rekomendasi berperan penting: memberikan saran yang dipersonalisasi berdasarkan histori penilaian dan preferensi pengguna. Penerapan teknologi ini tidak hanya meningkatkan pengalaman membaca, tetapi juga membantu penerbit dan toko buku daring dalam meningkatkan eksposur buku serta mendorong pertumbuhan penjualan.

Proyek ini bertujuan untuk membangun sistem rekomendasi buku yang berbasis data historis rating pengguna terhadap buku. Dengan memanfaatkan dataset dari Kaggle [3], proyek ini mengembangkan model rekomendasi yang mampu memberikan saran buku yang relevan dan bermanfaat bagi pengguna. Implementasi sistem seperti ini diharapkan dapat menjadi bagian dari solusi peningkatan literasi serta mendukung transformasi digital dalam industri perbukuan.

### Referensi:
[1] J. Ben Schafer, Joseph A. Konstan, and John Riedl, “E-Commerce Recommendation Applications,” *Data Mining and Knowledge Discovery*, vol. 5, no. 1-2, pp. 115–153, 2001.  
[2] J. Bennett and S. Lanning, “The Netflix Prize,” in *Proceedings of KDD Cup and Workshop*, 2007.  
[3] A. Nican, “Book Recommendation Dataset,” *Kaggle*, [Online]. Available: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

## Business Understanding

Sistem rekomendasi merupakan bagian penting dalam mempersonalisasi pengalaman pengguna di berbagai platform digital, termasuk platform literasi dan penjualan buku. Dengan memanfaatkan data historis berupa rating pengguna terhadap buku, sistem dapat memberikan rekomendasi buku lain yang sesuai dengan preferensi pengguna, bahkan jika buku tersebut belum pernah dibaca sebelumnya.

Pada bagian ini, akan dijelaskan proses klarifikasi masalah, tujuan dari proyek, serta pendekatan solusi yang digunakan untuk membangun sistem rekomendasi buku.

---

### Problem Statements

1. **Bagaimana mengidentifikasi pola preferensi pengguna terhadap buku berdasarkan data rating yang tersedia?**

2. **Bagaimana menyarankan buku yang belum pernah dibaca oleh pengguna, namun memiliki kemungkinan besar untuk disukai, berdasarkan kesamaan preferensi antar pengguna?**

3. **Bagaimana menyajikan hasil eksplorasi data dalam bentuk visualisasi untuk mendukung proses analisis dan pemodelan sistem rekomendasi?**

---

### Goals

1. **Mengolah data rating pengguna untuk memahami distribusi preferensi dan interaksi antar pengguna dan buku.**

2. **Membangun sistem rekomendasi berbasis collaborative filtering yang mampu menyarankan buku sesuai dengan selera pengguna yang memiliki pola interaksi serupa.**

3. **Menyajikan hasil eksplorasi data dalam bentuk visualisasi (seperti bar plot) untuk memberikan wawasan mengenai distribusi rating, usia pengguna, dan buku paling populer.**

---

### Solution Statements

Untuk mencapai tujuan yang telah ditentukan, proyek ini menggunakan pendekatan berikut:

1. **Neural Collaborative Filtering (NCF)**  
   Sistem rekomendasi ini menggunakan representasi embedding untuk memetakan pengguna dan buku ke dalam ruang vektor berdimensi rendah, lalu menghitung kemiripan atau interaksi antar keduanya menggunakan operasi dot product. Model ini dibangun dengan TensorFlow menggunakan layer `Embedding` untuk user dan item, serta dioptimasi untuk memprediksi rating dengan pendekatan regresi berbasis sigmoid.

   Keunggulan pendekatan ini adalah kemampuannya menangkap hubungan non-linear antara user dan item, serta fleksibilitas dalam pengembangan model yang lebih kompleks di masa depan.

2. **Visualisasi Eksploratif untuk Mendukung Rekomendasi**  
   Sebelum membangun model, dilakukan eksplorasi data melalui visualisasi seperti *bar plot* terhadap rating buku, usia pengguna, dan sebaran interaksi. Visualisasi ini membantu memahami distribusi data dan mendukung proses pengambilan keputusan dalam pemodelan.

Melalui pendekatan ini, sistem rekomendasi yang dibangun diharapkan mampu memberikan hasil yang relevan dan akurat berdasarkan preferensi pengguna yang tersirat dalam data rating.

## Data Understanding

Data Understanding merupakan tahap awal yang krusial dalam pengembangan proyek machine learning maupun data science. Tahap ini bertujuan untuk memahami isi, struktur, serta kualitas data yang akan dianalisis. Pada proyek ini, proses Data Understanding dilakukan melalui beberapa tahapan, yaitu:

1. Melakukan **load dataset** dan **mengubah nama kolom** agar lebih konsisten dan mudah dianalisis.
2. Melakukan **univariate exploratory data analysis** terhadap masing-masing dataset.
3. Melakukan **visualisasi data** untuk melihat distribusi dan pola data.
4. Mengelompokkan data berdasarkan **user yang memberikan rating terbanyak**.
5. Melakukan proses **penggabungan dataset**.
6. Mengelompokkan data berdasarkan **judul buku dengan jumlah rating terbanyak**.

Dataset ini diambil dari [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset), yang merupakan salah satu dataset rekomendasi buku terkenal. Dataset ini terdiri atas tiga file utama, yaitu:
- `Users.csv`
- `Ratings.csv`
- `Books.csv`

---

### Penjelasan Variabel

#### 1. Users

Dataset ini berisi informasi pengguna yang meliputi user_id, lokasi, dan usia.

| Kolom      | Deskripsi                                                                 |
|------------|---------------------------------------------------------------------------|
| `User_id`  | ID unik yang merepresentasikan masing-masing pengguna.                   |
| `Location` | Lokasi geografis pengguna (negara/kota/kode pos).                        |
| `Age`      | Usia pengguna (dapat berisi nilai kosong atau tidak valid).              |

---

#### 2. Ratings

Dataset ini merepresentasikan rating atau penilaian yang diberikan pengguna terhadap buku tertentu.

| Kolom     | Deskripsi                                                             |
|-----------|----------------------------------------------------------------------|
| `User_id` | ID pengguna yang memberi rating.                                     |
| `ISBN`    | ID buku yang dinilai.                                                |
| `Rating`  | Nilai rating yang diberikan (rentang 0–10; 0 bisa berarti tidak eksplisit). |

---

#### 3. Books

Dataset ini mencakup detail informasi buku yang ada di sistem.

| Kolom              | Deskripsi                                                                   |
|--------------------|------------------------------------------------------------------------------|
| `ISBN`             | Nomor unik identifikasi buku.                                               |
| `Title`            | Judul buku.                                                                 |
| `Author`           | Nama penulis buku.                                                          |
| `Year`             | Tahun publikasi buku.                                                       |
| `Publisher`        | Nama penerbit buku.                                                         |
| `Image-URL-S/M/L`  | URL gambar cover buku (kecil, sedang, besar) — dihapus karena tidak digunakan. |

---
### Data Quality & Preprocessing

- **Users**:
  - Missing values ditemukan pada `Age`.
  - Diisi dengan **modus** (nilai terbanyak).
- **Books**:
  - 2 nilai kosong pada `Year` → baris dihapus.
  - 3 nilai kosong pada kolom gambar → kolom `Image-URL` dihapus seluruhnya.
- **Ratings**:
  - Tidak ada missing values.
- **Duplikasi**: Tidak ditemukan pada ketiga dataset.

---

### Visualisasi Data (EDA)
Untuk memperkuat pemahaman terhadap karakteristik dataset, berikut adalah beberapa visualisasi yang dilakukan:
#### 1. Top Contributors in Book Dataset

**Insight**:
- Tahun terbit terbanyak: 2002, disusul 2001 dan 2000.
- Penerbit terbanyak:
  1. Harlequin
  2. Silhouette
  3. Pocket Books
  4. Ballantine Books
- Penulis terbanyak:
  1. Agatha Christie
  2. William Shakespeare
  3. Ann M. Martin
  4. Stephen King

---
#### 2. Distribution of Book Ratings

**Insight**:
- Rating 0 mendominasi (>700.000 entri) → menunjukkan **rating implisit**.
- Rating eksplisit (1–10) didominasi nilai tinggi:
  - Tertinggi pada rating 8, 10, dan 7.
  - Rating rendah (1–4) sangat sedikit.
- Pengguna cenderung hanya memberi rating pada buku yang mereka sukai.
- Ditemukan **20 pengguna paling aktif** dengan jumlah rating terbanyak, yang signifikan dalam membentuk sistem rekomendasi.
- Rata-rata rating per pengguna memberikan gambaran preferensi individual.

---
#### 3. 20 top location of the users
**Insight**:
- Mayoritas pengguna berasal dari negara-negara berbahasa Inggris.
- Lokasi pengguna terbanyak:
  - London, England, United Kingdom (>2.500 pengguna)
  - Toronto, Ontario (Canada)
  - Sydney, New South Wales (Australia)
  - Kota-kota besar di AS seperti Portland, Chicago, Seattle, New York, dan San Francisco.
  - Eropa: Madrid (Spanyol), Berlin (Jerman), Milano (Italia).
- Ini menunjukkan dominasi pengguna dari Amerika Utara, Eropa Barat, dan Australia — penting untuk mempertimbangkan keragaman budaya dalam sistem rekomendasi.

---


### Data Merging & Popular Books

- Dataset **Ratings** digabungkan dengan **Books** berdasarkan `ISBN`, menghasilkan lebih dari **1 juta baris data**.
- Setiap interaksi kini dilengkapi:
  - Judul buku
  - Penulis
  - Tahun terbit
  - Penerbit

**Insight**:
- Ditemukan **20 buku paling populer** (paling banyak dinilai pengguna).
- Buku-buku ini:
  - Memiliki **tingkat popularitas tinggi**.
  - Dapat digunakan sebagai prioritas dalam sistem rekomendasi.
  - Rata-rata rating buku mencerminkan **kualitas penerimaan** dari pembaca.

---


