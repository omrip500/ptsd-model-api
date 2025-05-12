from flask import Flask, request, jsonify
import tempfile
import os
import subprocess
import base64
import json

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        print("📥 קיבלנו בקשת /analyze")

        # הדפסת הקבצים שהתקבלו
        print("🔍 request.files:", request.files)

        image_file = request.files.get("image")
        yolo_file = request.files.get("yolo")

        if not image_file or not yolo_file:
            print("❌ קובץ image או yolo חסר")
            return jsonify({"error": "Missing image or yolo file"}), 400

        # שמירה זמנית
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_image, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_yolo:

            image_path = temp_image.name
            yolo_path = temp_yolo.name

            image_file.save(image_path)
            yolo_file.save(yolo_path)

        print(f"✅ נשמרו קבצים זמניים:\nImage: {image_path}\nYOLO: {yolo_path}")

        # הרצת סקריפט פייתון
        script_path = os.path.join(os.path.dirname(__file__), "analyze.py")
        print(f"🚀 מריץ סקריפט: {script_path}")

        result = subprocess.run(
            ["python3", script_path, image_path, yolo_path],
            capture_output=True,
            text=True
        )

        print("📤 stdout:")
        print(result.stdout)
        print("📛 stderr:")
        print(result.stderr)

        if result.returncode != 0:
            return jsonify({"error": result.stderr}), 500

        output = json.loads(result.stdout)

        # קריאת תמונות והמרה ל־base64
        with open(output["annotatedImagePath"], "rb") as f:
            annotated_b64 = base64.b64encode(f.read()).decode("utf-8")

        with open(output["convertedOriginalPath"], "rb") as f:
            original_b64 = base64.b64encode(f.read()).decode("utf-8")

        # ניקוי קבצים זמניים
        os.remove(image_path)
        os.remove(yolo_path)
        os.remove(output["annotatedImagePath"])
        os.remove(output["convertedOriginalPath"])

        print("✅ עיבוד הסתיים בהצלחה")

        return jsonify({
            "annotated_image_base64": annotated_b64,
            "converted_original_base64": original_b64,
            "summary": output["summary"]
        })

    except Exception as e:
        print("❌ שגיאה כללית:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6000))
    print(f"🚀 מריץ Flask על פורט {port}")
    app.run(debug=False, host="0.0.0.0", port=port)
