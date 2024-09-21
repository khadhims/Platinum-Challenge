from flask import Flask, request, json
from flask_restful import Api, Resource
from flask_swagger_ui import get_swaggerui_blueprint
from cleansing import preprocess, clean_csv
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle  # Import pickle untuk memuat model
import pandas as pd
import numpy as np

app = Flask(__name__)
api = Api(app)

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
      'app_name': "Sentiment Analisis API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Load your trained models and feature extractors here
with open('../pickle/model_MLPClassifier.p', 'rb') as f:
    mlp_model = pickle.load(f)

with open('../pickle/model_LSTM.p', 'rb') as f:
    lstm_model = pickle.load(f)

with open('../pickle/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# with open('../x_pad_sequences.pickle', 'rb') as f:
#     pad_sequences = pickle.load(f)

# Assuming you also saved your feature extraction steps like TF-IDF or word embedding methods
with open('../pickle/feature_tfidf_stp.p', 'rb') as f:
    feature_extractor = pickle.load(f)

class LstmText(Resource):
  def post(self):
    try:
      text = request.form['text']
      processed_text = preprocess(text)

      # Ekstrak fitur
      tok = tokenizer.texts_to_sequences([processed_text])
      guess = pad_sequences(tok, maxlen=77)
      # Prediksi menggunakan model LSTM
      prediction = lstm_model.predict(guess)
      sentiment_idx = int(np.argmax(prediction[0]))

      sentiments = ['Negative', 'Neutral', 'Positive']
      sentiment_label = sentiments[sentiment_idx]
      highest_prob = float(prediction[0][sentiment_idx])
      response = {
        'analysed_text': processed_text,
        'prediction': highest_prob,
        'sentiment': sentiment_label
      }

      return json.dumps(response), 200
    except Exception as e:
      print(f"Error in LSTM_Text: {e}")
      return json.dumps({'error': str(e)}), 500

class MlpText(Resource):
  def post(self):
    try:
      text = request.form['text']
      processed_text = preprocess(text)

      # Ekstrak fitur
      features = feature_extractor.transform([processed_text])
      # Prediksi menggunakan model MLP
      prediction = mlp_model.predict_proba(features)

      sentiment_idx = np.argmax(prediction[0])

      sentiments = ['Negative', 'Neutral', 'Positive']
      sentiment_label = sentiments[sentiment_idx]
      # highest_prob = float(prediction[0][sentiment_idx])

      response = {
        'analysed_text': processed_text,
        'prediction': float(prediction[0][sentiment_idx]),
        'sentiment': sentiment_label
      }      

      return json.dumps(response), 200
    except Exception as e:
      print(f"Error in MLP_Text: {e}")
      return json.dumps({'error': str(e)}), 500
  
class MlpCsv(Resource):
  def post(self):
    if 'file' not in request.files:
      return json.dumps({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
      return json.dumps({"error": "No selected file"}), 400
    if file:
      filename = secure_filename(file.filename)
      upload_dir = os.path.join(os.getcwd(), 'uploads')
      download_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
            
      os.makedirs(upload_dir, exist_ok=True)
            
      file_path = os.path.join(upload_dir, filename)
      file.save(file_path)
      try:
        df = clean_csv(file_path)
        features = feature_extractor.transform(df['cleaned_text'])

        # Prediksi menggunakan model MLPClassifier
        predictions = mlp_model.predict_proba(features)
        sentiments = ['Negative', 'Neutral', 'Positive']

        df['prediction'] = [sentiments[np.argmax(pred)] for pred in predictions]
        analysed_file_path = os.path.join(download_dir, f'analysed_{filename}')

        df.to_csv(analysed_file_path, index=False)

        response = {
          'message': 'File analysed, predictions made, and saved',
          'file_path': analysed_file_path
        }
            

        return json.dumps(response), 200
      except KeyError as e:
        print(f"Error processing CSV: {e}")
        return json.dumps({'error': str(e)}), 500
      
class LstmCsv(Resource):
  def post(self):
    if 'file' not in request.files:
      return json.dumps({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
      return json.dumps({"error": "No selected file"}), 400
    if file:
      filename = secure_filename(file.filename)
      upload_dir = os.path.join(os.getcwd(), 'uploads')
      download_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
            
      os.makedirs(upload_dir, exist_ok=True)
            
      file_path = os.path.join(upload_dir, filename)
      file.save(file_path)
      try:
        df = clean_csv(file_path)

        # Feature Extracting
        texts = df['cleaned_text'].tolist()
        tok = tokenizer.texts_to_sequences(texts)
        guess = pad_sequences(tok, maxlen=77)
        print(f"Debug - Type of guess: {type(guess)}")
        
        # Prediksi menggunakan model LSTM
        predictions = lstm_model.predict(guess)

        sentiments = ['Negative', 'Neutral', 'Positive']
        df['prediction'] = [sentiments[np.argmax(pred)] for pred in predictions]

        analysed_file_path = os.path.join(download_dir, f'analysed_{filename}')
        response = {
          'message'  : 'File analysed, predictions made, and saved',
          'file_path': analysed_file_path
        }
            
        df.to_csv(analysed_file_path, index=False)

        return json.dumps(response), 200
      except KeyError as e:
        print(f"Error processing CSV: {e}")
        return json.dumps({'error': str(e)}), 500

api.add_resource(LstmCsv, '/LSTM_CSV')
api.add_resource(MlpCsv, '/MLP_CSV')
api.add_resource(LstmText, '/LSTM_text')
api.add_resource(MlpText, '/MLP_text')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
