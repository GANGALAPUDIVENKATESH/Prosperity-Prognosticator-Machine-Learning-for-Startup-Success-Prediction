from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/random_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        funding_rounds = float(request.form["funding_rounds"])
        funding_amount = float(request.form["funding_amount"])
        market_size = float(request.form["market_size"])
        team_size = float(request.form["team_size"])
        years_active = float(request.form["years_active"])
        revenue_growth = float(request.form["revenue_growth"])

        input_data = np.array([[funding_rounds,
                                funding_amount,
                                market_size,
                                team_size,
                                years_active,
                                revenue_growth]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        result = "Acquired / Successful" if prediction == 1 else "Closed / Failed"

        return render_template("result.html", result=result)

    except Exception as e:
        return render_template("result.html", result="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
