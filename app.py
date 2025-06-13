import os
import numpy as np
from flask import Flask, render_template, request
import xgboost as xgb
import logging


# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model XGBoost
model = None # Inisialisasi model di luar blok try-except
try:
    # Gunakan metode load_model bawaan XGBoost untuk model yang disimpan sebagai JSON
    model_path = os.path.join(app.root_path, 'model', 'xgb_model.json')
    
    # Periksa apakah file model ada sebelum mencoba memuatnya
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan di: {model_path}")

    # Buat instance XGBClassifier kosong, lalu muat modelnya.
    # Ini penting agar atribut yang diperlukan oleh versi 2.1.4 tersedia.
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    logger.info(f"✅ Model XGBoost berhasil dimuat dari: {model_path}")
    logger.info(f"Versi XGBoost yang terinstal: {xgb.__version__}")
    # Dengan format JSON, seharusnya tidak ada lagi masalah 'get_params'.
    
except FileNotFoundError as e:
    logger.error(f"❌ Kesalahan Pemuatan Model: {e}")
    # Jika model tidak ditemukan, aplikasi tidak dapat berfungsi, hentikan eksekusi.
    raise SystemExit(f"Aplikasi tidak dapat dimulai: {e}. Pastikan model/xgb_model.json ada di direktori 'model/'.")
except Exception as e:
    logger.error(f"❌ Kesalahan Pemuatan Model: {str(e)}", exc_info=True)
    # Untuk kesalahan lain saat memuat model, hentikan eksekusi.
    raise SystemExit("Aplikasi tidak dapat dimulai karena model gagal dimuat. Periksa log untuk detail lebih lanjut.")

# Definisi fitur-fitur yang digunakan oleh model
FEATURES = [
    'CycleWithPeakorNot', 'EstimatedDayofOvulation', 'FirstDayofHigh',
    'LengthofLutealPhase', 'NumberofDaysofIntercourse', 'TotalNumberofHighDays',
    'TotalHighPostPeak', 'TotalNumberofPeakDays', 'TotalDaysofFertility',
    'TotalFertilityFormula', 'LengthofMenses', 'UnusualBleeding',
    'StressScore', 'DietScore', 'ReproHealthScore', 'IntercourseInFertileWindow'
]

# Label prediksi untuk output model
PREDICTION_LABELS = {
    0: "Siklus Sedang",
    1: "Siklus Panjang",
    2: "Siklus Pendek",
    3: "Tidak Normal"
}

# Fungsi untuk memvalidasi input dari formulir
def validate_inputs(form_data):
    validated = {}
    errors = []
    
    # Aturan validasi untuk setiap bidang input
    validation_rules = {
        'cycle_length': {'min': 15, 'max': 50, 'type': 'float', 'name': 'Panjang Siklus'},
        'period_length': {'min': 2, 'max': 10, 'type': 'float', 'name': 'Lama Menstruasi'},
        'age': {'min': 15, 'max': 50, 'type': 'float', 'name': 'Usia'},
        'CycleWithPeakorNot': {'type': 'binary', 'name': 'Siklus dengan Peak'},
        'EstimatedDayofOvulation': {'min': 1, 'max': 30, 'type': 'float', 'name': 'Perkiraan Ovulasi'},
        'FirstDayofHigh': {'min': 1, 'max': 30, 'type': 'float', 'name': 'Hari Pertama Tinggi'},
        'LengthofLutealPhase': {'min': 1, 'max': 20, 'type': 'float', 'name': 'Panjang Fase Luteal'},
        'NumberofDaysofIntercourse': {'min': 0, 'max': 30, 'type': 'int', 'name': 'Jumlah Hari Berhubungan'},
        'TotalNumberofHighDays': {'min': 0, 'max': 30, 'type': 'int', 'name': 'Total Hari Tinggi'},
        'TotalHighPostPeak': {'min': 0, 'max': 30, 'type': 'int', 'name': 'Total High Post Peak'},
        'TotalNumberofPeakDays': {'min': 0, 'max': 30, 'type': 'int', 'name': 'Total Hari Peak'},
        'TotalDaysofFertility': {'min': 0, 'max': 30, 'type': 'int', 'name': 'Total Hari Subur'},
        'TotalFertilityFormula': {'min': 0, 'max': 30, 'type': 'int', 'name': 'Total Fertility Formula'},
        'LengthofMenses': {'min': 1, 'max': 10, 'type': 'int', 'name': 'Panjang Menstruasi'},
        'UnusualBleeding': {'type': 'binary', 'name': 'Perdarahan Tidak Biasa'},
        'DietScore': {'min': 1, 'max': 5, 'type': 'int', 'name': 'Skor Diet'},
        'ReproHealthScore': {'min': 1, 'max': 5, 'type': 'int', 'name': 'Skor Kesehatan Reproduksi'},
        'StressScore': {'min': 1, 'max': 5, 'type': 'int', 'name': 'Skor Stres'},
        'IntercourseInFertileWindow': {'type': 'binary', 'name': 'Hubungan di Masa Subur'}
    }

    # Iterasi melalui setiap aturan validasi dan menerapkan validasi
    for field, rule in validation_rules.items():
        try:
            value = form_data.get(field, '').strip()
            if not value:
                errors.append(f"{rule['name']} harus diisi")
                continue
                
            # Konversi nilai ke tipe data yang sesuai
            if rule['type'] == 'float':
                value = float(value)
            elif rule['type'] == 'int':
                value = int(value)
            elif rule['type'] == 'binary': # Validasi untuk input biner (0 atau 1)
                value = int(value)
                if value not in [0, 1]:
                    errors.append(f"{rule['name']} harus 0 atau 1")
                    continue
                    
            # Validasi rentang nilai (min/max)
            if 'min' in rule and value < rule['min']:
                errors.append(f"{rule['name']} minimal {rule['min']}")
            if 'max' in rule and value > rule['max']:
                errors.append(f"{rule['name']} maksimal {rule['max']}")
                
            validated[field] = value
        except ValueError:
            # Tangani kesalahan konversi tipe data (misalnya, input non-angka)
            errors.append(f"{rule['name']} harus berupa angka yang valid")
    
    # Jika ada kesalahan validasi, gabungkan dan berikan sebagai ValueError
    if errors:
        raise ValueError("<br>".join(errors))
    
    return validated

# Rute untuk halaman utama
@app.route('/')
def home():
    return render_template('home.html', 
                         page_title='Home',
                         active_page='home')

# Rute untuk halaman "Tentang Kami"
@app.route('/about')
def about():
    return render_template('about.html', 
                         page_title='About',
                         active_page='about')

# Rute untuk halaman "Tim Kami"
@app.route('/team')
def team():
    return render_template('team.html', 
                         page_title='Our Team',
                         active_page='team')

# Rute untuk formulir prediksi dan pemrosesan POST
@app.route('/form', methods=['GET', 'POST'])
def form():
    # Pastikan model telah dimuat sebelum mencoba melakukan prediksi
    if model is None:
        logger.error("Model tidak dimuat saat mencoba akses form.")
        return render_template('form.html',
                               page_title='Form Prediksi',
                               active_page='form',
                               error="Sistem prediksi tidak tersedia. Mohon coba lagi nanti.")

    if request.method == 'POST':
        try:
            # Validasi data input dari formulir
            data = validate_inputs(request.form)
            logger.info(f"Data valid: {data}")
            
            # Siapkan input untuk model prediksi
            input_features = [data.get(f, 0) for f in FEATURES]
            input_array = np.array(input_features).reshape(1, -1) 
            
            # Lakukan prediksi menggunakan model XGBoost
            # Karena model dimuat sebagai xgb.XGBClassifier, metode predict() dan predict_proba() tersedia
            if isinstance(model, xgb.XGBClassifier):
                prediction = model.predict(input_array)[0]
                proba = float(np.max(model.predict_proba(input_array))) * 100
                prediction_label = PREDICTION_LABELS.get(prediction, "Tidak Diketahui")
            else:
                logger.error("Model yang dimuat bukan instance XGBoost classifier.")
                raise TypeError("Model yang dimuat bukan model XGBoost yang valid untuk prediksi. Tipe model: " + str(type(model)))
            
            # Tentukan kategori siklus berdasarkan panjang siklus
            cycle_length = data['cycle_length']
            if cycle_length < 21:
                cycle_category = "Siklus Pendek"
            elif 21 <= cycle_length <= 35:
                cycle_category = "Siklus Sedang"
            else:
                cycle_category = "Siklus Panjang"
            
            # Pemetaan skor kesehatan dan stres ke label yang mudah dibaca
            health_mapping = {
                1: "Buruk",
                2: "Buruk",
                3: "Normal",
                4: "Baik",
                5: "Baik"
            }
            
            stress_mapping = {
                1: "Rendah",
                2: "Sedang",
                3: "Normal",
                4: "Tinggi",
                5: "Sangat Tinggi"
            }
            
            # Siapkan data untuk ditampilkan di halaman hasil
            display_data = {
                'cycle': {'label': 'Panjang Siklus', 'value': f"{data['cycle_length']} hari"},
                'cycle_category': {'label': 'Kategori Siklus', 'value': cycle_category},
                'period': {'label': 'Lama Menstruasi', 'value': f"{data['period_length']} hari"},
                'age': {'label': 'Usia', 'value': f"{data['age']} tahun"},
                'ovulation': {'label': 'Perkiraan Ovulasi', 'value': f"Hari ke-{data['EstimatedDayofOvulation']}"},
                'luteal': {'label': 'Panjang Fase Luteal', 'value': f"{data['LengthofLutealPhase']} hari"},
                'intercourse': {'label': 'Jumlah Hari Berhubungan', 'value': data['NumberofDaysofIntercourse']},
                'fertile_window': {'label': 'Hubungan di Masa Subur', 'value': 'Ya' if data['IntercourseInFertileWindow'] == 1 else 'Tidak'},
                'peak': {'label': 'Siklus dengan Peak', 'value': 'Ya' if data['CycleWithPeakorNot'] == 1 else 'Tidak'},
                'bleeding': {'label': 'Perdarahan Tidak Biasa', 'value': 'Ya' if data['UnusualBleeding'] == 1 else 'Tidak'},
                'diet': {'label': 'Skor Diet', 'value': health_mapping.get(data['DietScore'], "Tidak Diketahui")},
                'repro_health': {'label': 'Skor Kesehatan Reproduksi', 'value': health_mapping.get(data['ReproHealthScore'], "Tidak Diketahui")},
                'stress': {'label': 'Skor Stres', 'value': stress_mapping.get(data['StressScore'], "Tidak Diketahui")}
            }
            
            # Tentukan kelas CSS untuk warna prediksi
            if "Siklus Sedang" in prediction_label:
                prediction_class = 'text-green-600'
            elif "Siklus Pendek" in prediction_label or "Siklus Panjang" in prediction_label:
                prediction_class = 'text-yellow-600'
            else:
                prediction_class = 'text-red-600'
            
            # Render halaman hasil dengan data prediksi
            return render_template(
                'result.html',
                page_title='Hasil Prediksi',
                active_page='result',
                prediction=prediction_label,
                probability=proba,
                details=display_data,
                prediction_class=prediction_class
            )
            
        except ValueError as e:
            # Tangani kesalahan validasi input
            return render_template('form.html',
                                page_title='Form Prediksi',
                                active_page='form',
                                error=str(e))
        except Exception as e:
            # Tangani kesalahan sistem lainnya saat pengiriman formulir
            logger.error(f"Kesalahan saat pengiriman formulir: {str(e)}", exc_info=True)
            return render_template('form.html',
                                page_title='Form Prediksi',
                                active_page='form',
                                error="Terjadi kesalahan sistem saat memproses input Anda. Silakan coba lagi.")
    
    # Render halaman formulir kosong untuk permintaan GET
    return render_template('form.html',
                         page_title='Form Prediksi',
                         active_page='form')

# Penanganan error 404 (Halaman Tidak Ditemukan)
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html',
                         page_title='Halaman Tidak Ditemukan'), 404

# Penanganan error 500 (Kesalahan Server Internal)
@app.errorhandler(500)
def server_error(e):
    return render_template('500.html',
                         page_title='Kesalahan Server'), 500

# Baris ini biasanya dikomentari saat di-deploy ke cPanel dengan Passenger/WSGI
if __name__ == '__main__':
    app.run(debug=True)
