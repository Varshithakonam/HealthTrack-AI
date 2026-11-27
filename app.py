from flask import Flask, render_template, request, session
import joblib
import os

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY", "development-only-change-this"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "health_model.pkl")

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():

    error = None

    if request.method == "POST":

        symptoms = request.form.get("symptoms", "").strip()

        if not symptoms:
            error = "Please enter at least one symptom"
            return render_template("index.html", error=error)

        try:
            probabilities = model.predict_proba([symptoms])[0]
            classes = model.classes_

            predictions = sorted(
                zip(classes, probabilities),
                key=lambda item: item[1],
                reverse=True
            )[:3]

            results = [
                {
                    "condition": condition,
                    "confidence": round(float(score) * 100, 2)
                }
                for condition, score in predictions
            ]

            if not results:
                raise ValueError("No predictions available")

            session["results"] = results

            return render_template(
                "results.html",
                results=results
            )

        except Exception:
            error = "Unable to predict conditions at this time"

    return render_template("index.html", error=error)


@app.route("/save")
def save_results():

    results = session.get("results")

    if not results:
        return "No results available to save.", 404

    from flask import make_response
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Potential Condition", "Model Score (%)"])

    for item in results:
        writer.writerow([
            item["condition"],
            item["confidence"]
        ])

    response = make_response(output.getvalue())

    response.headers["Content-Disposition"] = (
        "attachment; filename=healthtrack_results.csv"
    )

    response.headers["Content-Type"] = "text/csv"

    return response


if __name__ == "__main__":
    app.run(debug=True)