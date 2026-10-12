from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from edge_processor import EdgeProcessor

app = Flask(__name__, template_folder="templates")
CORS(app)  # Enables cross-origin requests from web browsers

# Initialize edge processor
processor = EdgeProcessor(blur_threshold=100.0, phash_threshold=5, entropy_threshold=1.5)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_frame():
    if "file" not in request.files:
        return jsonify({"status": "ERROR", "reason": "No image file provided."}), 400

    file = request.files["file"]
    species_name = request.form.get("species_name", "Bengal Tiger")
    blur_threshold = float(request.form.get("blur_threshold", 100))

    file_bytes = file.read()

    status, blur_score, reason = processor.process_frame(
        file_bytes=file_bytes,
        target_species=species_name,
        blur_threshold=blur_threshold
    )

    return jsonify({
        "status": status,
        "blur_score": blur_score,
        "reason": reason
    }), 200

if __name__ == "__main__":
    print("Serving Flask app at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)