
import os
import tempfile

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from edge_processor import EdgeProcessor

app = Flask(__name__, template_folder="templates")
CORS(app)

processor = EdgeProcessor(
    blur_threshold=100.0,
    phash_threshold=5,
    entropy_threshold=1.5
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_frame():
    if "file" not in request.files:
        return jsonify({
            "status": "ERROR",
            "reason": "No image file provided."
        }), 400

    uploaded_file = request.files["file"]

    suffix = os.path.splitext(uploaded_file.filename or "")[1] or ".jpg"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp_file:
            uploaded_file.save(temp_file)
            temp_path = temp_file.name

        status, blur_score, reason = processor.process_frame(temp_path)

        return jsonify({
            "status": status,
            "blur_score": blur_score,
            "reason": reason
        }), 200

    except Exception:
        app.logger.exception("Image processing failed")
        return jsonify({
            "status": "ERROR",
            "reason": "Image processing failed. Check server logs."
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
```
