# Face Detection & Recognition — Computer Vision Project

Nama : Athaya Naura Khalilah  
NIM : 23/512716/PA/21899

---

## Deskripsi
Project ini merupakan implementasi face detection dan face recognition menggunakan metode Eigenface.  
Model ini dikembangkan menggunakan dataset gambar wajah dan dioptimalkan untuk pengenalan wajah secara real-time melalui webcam.

Semua proses dilakukan secara lokal tanpa bergantung pada layanan cloud.

---

## Tools & Library
- Python 3.11
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Pickle

---
## Step-by-Step Menjalankan Project

1. Download atau clone repository ini ke komputer kamu menggunakan VSCode:
```bash
git clone https://github.com/username/FaceDetection-Recognition.git
cd FaceDetection-Recognition
```

2. Jika ada notifikasi extension di VSCode, install sesuai rekomendasi.

3. Pastikan pip sudah tersedia. Untuk memastikan atau memperbarui pip, jalankan:
```bash
python -m pip install --upgrade pip
```

4. Install semua library yang diperlukan:
```bash
pip install numpy opencv-python scikit-learn matplotlib
```

5. Pastikan folder `images/` sudah ada dan berisi dataset wajah, tersusun dalam folder berdasarkan nama individu.

6. Jalankan script utama dengan perintah:
```bash
python face_detection_recognition_cv.py
```

7. Program akan otomatis:
- Melakukan training model jika belum ada `eigenface_pipeline.pkl`
- Menyimpan model hasil training ke dalam file
- Mengaktifkan webcam dan melakukan real-time face recognition

---

## Fitur
- Face Detection menggunakan Haarcascade dari OpenCV.
- Face Recognition menggunakan PCA + SVM (Eigenface Method).
- Support real-time detection via webcam.
- Training otomatis jika model belum tersedia.

---

## Catatan
- Jika ingin menambahkan dataset baru, cukup tambahkan gambar wajah ke dalam folder `images/` dengan struktur sesuai nama individu (format yang sesuai dengan dataset lainnya).
- Model `eigenface_pipeline.pkl` yang sudah disimpan dapat langsung digunakan atau diupdate dengan data baru jika diperlukan.
- Apabila webcam bermasalah saat dijalankan, pastikan izin akses webcam aktif, atau coba ubah parameter index pada `cv2.VideoCapture()`.

---

## Contoh Output
- Visualisasi Eigenfaces setelah training.
- Real-time bounding box di wajah dengan nama prediksi dan skor.

(Tambahkan screenshot hasil deteksi di sini jika perlu)

---

# Selesai!

Jika terdapat kendala, pastikan semua path benar dan library sudah terinstall.  
Happy Coding! 🚀
