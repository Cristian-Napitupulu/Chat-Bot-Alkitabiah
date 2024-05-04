# Chat Bot Alkitabiah
Repository ini menyimpan program untuk Chat Bot Alkitabiah. \
Chat Bot Alkitabiah ini digunakan sebagai sarana konseling dalam perspektif agama Kristen.

# Installation
Proses instalasi dapat dilakukan dengan melakukan proses berikut.
1. Pada terminal, masukkan perintah:
> $ git clone https://github.com/Cristian-Napitupulu/Chat-Bot-Alkitabiah.git

2. Setelah selesai, pindah ke folder Chat-Bot-Alkitabiah dan install semua modul python yang diperlukan dengan perintah berikut:
> $ cd ./Chat-Bot-Alkitabiah

> $ pip install -r requirements.txt

3. Kemudian, jalankan program "olah_intents.py" pada folder "./data" untuk mengolah data intents.

4. Lalu, lakukan training untuk fine-tuning BERT model melalui program "train_bert.py".

5. Setelah proses fine-tuning selesai, deploy Chat-Bot-Alkitabiah dengan menjalankan program "app.py" pada folder "./src"

6. Chat-Bot-Alkitabiah akan di deploy dan dapat diakses melalui browser dengan menggunakan link berikut.
> http://127.0.0.1:5000
