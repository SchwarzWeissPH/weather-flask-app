from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np
import pymysql
from keras.models import load_model
from keras.utils import custom_object_scope
from tcn import TCN
from datetime import datetime
import logging
from threading import Thread

# --- CONFIGURATION ---
MODEL_PATH = 'tcnflask.keras'

DB_CONFIG = {
    'host': '118.139.162.228',
    'user': 'drei',
    'password': 'madalilangto',
    'database': 'signup'
}

# --- LOGGING CONFIG ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- INITIALIZE FLASK ---
app = Flask(__name__)

# Load the model once
model = None

def load_weather_model():
    global model
    if model is None:
        with custom_object_scope({'TCN': TCN}):
            model = load_model(MODEL_PATH)
        logger.info("[INFO] Model loaded.")
    return model

# --- DATABASE CONNECTION ---
def connect_to_database():
    try:
        conn = pymysql.connect(**DB_CONFIG, connect_timeout=60, read_timeout=60)
        return conn
    except Exception as e:
        logger.error(f"[ERROR] Database connection failed: {e}")
        return None

# --- PREDICTION FUNCTION ---
def run_prediction():
    try:
        model = load_weather_model()
        conn = connect_to_database()
        if not conn:
            logger.error("[ERROR] Database connection failed.")
            return

        rows = fetch_latest_weather_data(conn)
        if len(rows) < 24:
            logger.error("[ERROR] Not enough data for prediction.")
            return

        X_raw = np.array(rows).astype(float)
        X_input = X_raw.reshape(1, 24, 4)
        prediction = model.predict(X_input)[0]
        save_prediction_to_db(conn, prediction)
        logger.info(f"[INFO] Prediction made: {prediction}")

    except Exception as e:
        logger.error(f"[ERROR] {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

# --- HELPER FUNCTIONS ---
def fetch_latest_weather_data(conn):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT temperature, humidity, pressure 
        FROM weather_data_2 
        ORDER BY timestamp DESC 
        LIMIT 24
    """)
    rows = cursor.fetchall()
    cursor.close()
    return rows

def save_prediction_to_db(conn, prediction):
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        cursor.execute("""
            INSERT INTO signup_predictions 
            (predicted_value_1, predicted_value_2, predicted_value_3, timestamp)
            VALUES (%s, %s, %s, %s)
        """, (
            float(prediction[0]),
            float(prediction[1]),
            float(prediction[2]),
            timestamp
        ))
        conn.commit()
        logger.info("[INFO] Prediction saved to database.")
    except pymysql.MySQLError as e:
        logger.error(f"[ERROR] Saving prediction failed: {e}")
        conn.rollback()
    finally:
        cursor.close()

# --- ROUTES ---
@app.route('/predict', methods=['GET'])
def predict_weather():
    try:
        # Load model
        model = load_weather_model()

        # Connect to database
        conn = connect_to_database()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500

        # Fetch data
        rows = fetch_latest_weather_data(conn)

        # Auto-padding if rows are less than 24
        if len(rows) < 24:
            print(f"[WARN] Only {len(rows)} rows found, padding to 24...")
            last_row = rows[-1] if rows else (0, 0, 0)
            while len(rows) < 24:
                rows.append(last_row)

        # Prepare input
        rows = rows[::-1]  # Most recent last
        X_raw = np.array(rows).astype(float)
        X_raw_padded = np.concatenate([X_raw, np.zeros((X_raw.shape[0], 1))], axis=1)
        X_input = X_raw_padded.reshape(1, 24, 4)

        # Predict
        prediction = model.predict(X_input)[0]
        if prediction.ndim > 1:
            prediction = prediction.flatten()

        # Save to DB
        save_prediction_to_db(conn, prediction)

        # Return prediction as JSON
        return jsonify({
            'predicted_temperature': float(prediction[0]),
            'predicted_humidity': float(prediction[1]),
            'predicted_pressure': float(prediction[2]),
            'status': 'Prediction saved successfully'
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        if 'conn' in locals() and conn:
            conn.close()

# --- SCHEDULER SETUP ---
scheduler = BackgroundScheduler()
scheduler.add_job(func=run_prediction, trigger="interval", minutes=1)  # Run every minute
scheduler.start()

# --- MAIN ---
def start_flask_app():
    app.run(host='0.0.0.0', port=10000)

if __name__ == '__main__':
    # Start Flask in a separate thread to not block the scheduler
    flask_thread = Thread(target=start_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    # Keep the main thread alive to allow the scheduler to run
    while True:
        time.sleep(1)
