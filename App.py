from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Location': request.form['location'],
        'Condition': request.form['condition'],
        'Bedrooms': int(request.form['bedrooms']),
        'Bathrooms': int(request.form['bathrooms']),
        'Size_m2': float(request.form['size']),
        'Year_Built': int(request.form['year'])
    }

    df = pd.DataFrame([data])
    price = model.predict(df)[0]
    price = f"₦{price:,.2f}"
    return render_template('index.html', prediction=price)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

