# Irigasi Pintar

Dashboard Streamlit untuk menilai kebutuhan penyiraman cabai, tomat, dan selada pada 16 petak rumah kaca di Lembang. Python menjalankan inferensi GBDT saat pengguna mengubah masukan. Aplikasi memakai data historis dan masukan manual tanpa layanan API tambahan.

## Anggota Tim

Dikembangkan oleh:

- **Naail Safwan Saleh** — Ketua Tim
- **Juan Stevan Immanuel** — Anggota

## Menjalankan aplikasi

Gunakan Python 3.12 dari direktori yang berisi `app.py`.

```bash
python -m venv .venv
```

Aktifkan lingkungan dengan `.venv\Scripts\activate` di Windows atau `source .venv/bin/activate` di Linux dan macOS.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Link Streamlit Community Cloud

[Judul Link](https://contoh.com)

## Menggunakan dashboard

- **Konsol** menyediakan empat contoh kondisi. Pilih petak, sesuaikan pembacaan, lalu lihat peluang perlu siram, rekomendasi dan pembanding ambang tanah 35%. Buka kartu glosarium untuk membaca definisi.
- **16 Petak** menampilkan tiga kondisi historis. Pilih satu petak dan muat kondisinya ke Konsol untuk mencoba perubahan.
- **Evaluasi** menampilkan hasil uji, kurva metrik dan biaya kesalahan pada berbagai ambang. Ambang simulasi hanya memengaruhi perhitungan pada tab ini.

## Model dan data

Dashboard memakai pohon terlatih dari versi Editorial dengan ambang keputusan 0,23. Konsol memperkirakan riwayat kelembapan dari satu masukan, sedangkan pipeline pelatihan menghitung fitur dari urutan observasi historis. Pada periode uji 15–31 Oktober 2025, model mendeteksi 131 dari 175 kebutuhan siram, dengan 7 alarm keliru dan 44 kebutuhan terlewat. Kinerja di lokasi atau musim lain memerlukan pengujian lapangan.

| Berkas | Kegunaan |
| --- | --- |
| app.py | Halaman Streamlit, kontrol petak dan evaluasi |
| model.py dan model.json | Inferensi Python, parameter pohon dan kurva ambang |
| petak.json dan glosarium.json | Kondisi historis dan kartu istilah |
| raw.csv dan pipeline.py | Data sumber dan eksperimen pelatihan |
| requirements.txt | Dependensi aplikasi dan pelatihan |

Untuk mengulang eksperimen, jalankan `python pipeline.py`. Skrip menampilkan hasil evaluasi di terminal. Aplikasi memuat model tersimpan dari `model.json` secara terpisah.

