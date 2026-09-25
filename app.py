from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""
    error = ""

    if request.method == "POST":

        try:
            amount = float(request.form["amount"])
            transactions = int(request.form["transactions"])
            previous_fraud = int(request.form["previous_fraud"])
            distance = float(request.form["distance"])

            if amount <= 0:
                error = "Amount must be greater than 0."

            elif transactions <= 0:
                error = "Transactions must be greater than 0."

            elif previous_fraud not in [0, 1]:
                error = "Previous Fraud must be 0 or 1."

            elif distance < 0:
                error = "Distance cannot be negative."

            else:

                data = [[
                    amount,
                    transactions,
                    previous_fraud,
                    distance
                ]]

                prediction = model.predict(data)[0]

                probability = model.predict_proba(data)[0][1] * 100

                if probability >= 70:
                    risk = "HIGH"

                elif probability >= 40:
                    risk = "MEDIUM"

                else:
                    risk = "LOW"

                if prediction == 1:
                    result = (
                        f"FRAUD | Probability: "
                        f"{probability:.1f}% | Risk: {risk}"
                    )
                else:
                    result = (
                        f"NORMAL | Probability: "
                        f"{probability:.1f}% | Risk: {risk}"
                    )

        except ValueError:
            error = "Please enter valid numbers."

    return render_template(
        "index.html",
        result=result,
        error=error
    )


app.run(debug=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
