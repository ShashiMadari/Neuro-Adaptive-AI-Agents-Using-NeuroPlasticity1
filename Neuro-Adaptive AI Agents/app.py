from flask import Flask, request, jsonify, render_template
import os, traceback
from werkzeug.utils import secure_filename

# Import from your core system
from main8 import run_multi_agent_query

app = Flask(__name__)

# === Configuration ===
UPLOAD_FOLDER = "uploads"
STATIC_AUDIO_FOLDER = os.path.join("static", "audio")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_AUDIO_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "wav", "mp3", "txt", "pdf", "docx"}


# === Helper function ===
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# === Routes ===
@app.route("/")
def home():
    return render_template("index1.html")


@app.route("/run_agent", methods=["POST"])
def run_agent():
    try:
        text_input = request.form.get("text_input", "")
        agent_type = request.form.get("agent_type", "normal")

        image_file = request.files.get("image_file")
        audio_file = request.files.get("audio_file")

        image_path = audio_path = None

        # Save uploaded image
        if image_file and allowed_file(image_file.filename):
            image_filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            image_file.save(image_path)

        # Save uploaded audio (if user provides one)
        if audio_file and allowed_file(audio_file.filename):
            audio_filename = secure_filename(audio_file.filename)
            audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)
            audio_file.save(audio_path)

        # Run the multi-agent system
        result = run_multi_agent_query(
            agent_type=agent_type,
            image_path=image_path,
            audio_path=audio_path,
            text_input=text_input,
        )

        # ✅ If the model-generated audio file exists, make it browser-accessible
        if "audio_file" in result and os.path.exists(result["audio_file"]):
            audio_filename = os.path.basename(result["audio_file"])
            result["audio_file"] = f"/static/audio/{audio_filename}"

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        })


# === Main entry point ===
if __name__ == "__main__":
    app.run(debug=True)
