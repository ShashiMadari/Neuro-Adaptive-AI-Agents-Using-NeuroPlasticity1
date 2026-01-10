from flask import Flask, request, jsonify, render_template, redirect, url_for
import os, traceback, uuid
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
from pydub import AudioSegment

# Import from your core system
from main8 import run_multi_agent_query
from models.image_captioning import generate_caption
from models.speech_to_text import transcribe_audio

app = Flask(__name__)

# === Configuration ===
UPLOAD_FOLDER = "uploads"
STATIC_AUDIO_FOLDER = os.path.join("static", "audio")
PLOTS_FOLDER = os.path.join("static", "plots")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_AUDIO_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "wav", "mp3", "txt", "pdf", "docx"}

model = SentenceTransformer("all-MiniLM-L6-v2")

# === Helper ===
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def save_plot(fig):
    plot_filename = f"plot_{uuid.uuid4().hex}.png"
    plot_path = os.path.join(PLOTS_FOLDER, plot_filename)
    fig.savefig(plot_path)
    plt.close(fig)
    return f"/static/plots/{plot_filename}"


# === ROUTES ===
@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/start")
def start_system():
    return render_template("index2.html")

@app.route("/visualize")
def visualize_page():
    """Page to show 3D visualization (data passed via query params)."""
    plot_url = request.args.get("plot")
    image_caption = request.args.get("img_cap")
    audio_transcript = request.args.get("audio_cap")
    text_query = request.args.get("text_in")
    return render_template("visualize.html",
                           plot_url=plot_url,
                           image_caption=image_caption,
                           audio_transcript=audio_transcript,
                           text_query=text_query)


@app.route("/generate_3d", methods=["POST"])
def generate_3d():
    """Generates 3D modality space before reasoning"""
    try:
        text_input = request.form.get("text_input", "")
        image_file = request.files.get("image_file")
        audio_file = request.files.get("audio_file")

        image_path, audio_path = None, None

        # Save uploaded files
        if image_file and allowed_file(image_file.filename):
            image_path = os.path.join(UPLOAD_FOLDER, secure_filename(image_file.filename))
            image_file.save(image_path)

        
        if audio_file and allowed_file(audio_file.filename):
            audio_filename = secure_filename(audio_file.filename)
            audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
            audio_file.save(audio_path)

    # 🔄 Convert any audio format to mono PCM WAV
            try:
                audio = AudioSegment.from_file(audio_path)
                audio = audio.set_channels(1)  # convert to mono
                audio = audio.set_frame_rate(16000)  # standard speech rate
                pcm_path = os.path.splitext(audio_path)[0] + "_mono.wav"
                audio.export(pcm_path, format="wav")
                audio_path = pcm_path  # overwrite with converted path
            except Exception as e:
                print(f"⚠️ Audio conversion failed: {e}")

        # Generate caption, transcript, and use text input
        image_caption = generate_caption(image_path) if image_path else "No image provided"
        audio_transcript = transcribe_audio(audio_path) if audio_path else "No audio provided"
        text_query = text_input or "No text provided"

        # Encode modalities
        X = model.encode(image_caption)
        Y = model.encode(audio_transcript)
        Z = model.encode(text_query)

        embeddings = np.vstack([X, Y, Z])
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(embeddings)
        x, y, z = reduced[:, 0], reduced[:, 1], reduced[:, 2]

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(x[0], y[0], z[0], color="blue", s=80, label="Image")
        ax.scatter(x[1], y[1], z[1], color="green", s=80, label="Audio")
        ax.scatter(x[2], y[2], z[2], color="red", s=80, label="Text")
        ax.set_xlabel("Image (X)")
        ax.set_ylabel("Audio (Y)")
        ax.set_zlabel("Text (Z)")
        ax.set_title("3D Input Modality Space")
        ax.legend()

        plot_url = save_plot(fig)

        # Redirect to visualization page
        return jsonify({
            "redirect_url": url_for("visualize_page",
                                    plot=plot_url,
                                    img_cap=image_caption,
                                    audio_cap=audio_transcript,
                                    text_in=text_query)
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})


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

        result = run_multi_agent_query(
            agent_type=agent_type,
            image_path=image_path,
            audio_path=audio_path,
            text_input=text_input,
        )

        if "audio_file" in result and os.path.exists(result["audio_file"]):
            audio_filename = os.path.basename(result["audio_file"])
            result["audio_file"] = f"/static/audio/{audio_filename}"

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        })

@app.route("/visualize_reasoning", methods=["POST"])
def visualize_reasoning():
    """Generate 3D reasoning embedding graph for all three agents."""
    try:
        reasoning_normal = request.form.get("reasoning_normal", "")
        reasoning_blind = request.form.get("reasoning_blind", "")
        reasoning_mute = request.form.get("reasoning_mute", "")

        if not all([reasoning_normal, reasoning_blind, reasoning_mute]):
            return jsonify({"error": "Missing reasoning sentences."})

        # --- Compute embeddings and reduce to 3D ---
        f_normal = model.encode(reasoning_normal)
        f_blind = model.encode(reasoning_blind)
        f_mute = model.encode(reasoning_mute)

        embeddings = np.vstack([f_normal, f_blind, f_mute])
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(embeddings)
        x, y, z = reduced[:, 0], reduced[:, 1], reduced[:, 2]

        # --- Compute distances and closest agent ---
        dist_blind = float(np.linalg.norm(reduced[0] - reduced[1]))
        dist_mute = float(np.linalg.norm(reduced[0] - reduced[2]))
        closest = "Mute Agent" if dist_mute < dist_blind else "Blind Agent"

        # --- Plot 3D Reasoning Space ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # === 1️⃣ Normal Agent (Full modality reasoning) ===
        ax.scatter(x[0], y[0], z[0], s=180, color='blue', label='Normal Agent', edgecolor='k')
        ax.text(x[0], y[0], z[0]+0.03, "Normal", color='blue', fontsize=10, weight='bold')

        # === 2️⃣ Blind Agent (Audio–Text Plane → Y–Z) ===
        y_plane = np.linspace(-1, 1, 10)
        z_plane = np.linspace(-1, 1, 10)
        Yb, Zb = np.meshgrid(y_plane, z_plane)
        Xb = np.full_like(Yb, np.mean(x))  # No visual input
        ax.plot_surface(Xb, Yb, Zb, color='green', alpha=0.2)
        ax.scatter(x[1], y[1], z[1], s=150, color='green', label='Blind Agent', edgecolor='k')
        ax.text(x[1], y[1], z[1]+0.03, "Blind", color='green', fontsize=10, weight='bold')

        # === 3️⃣ Mute Agent (Image–Text Plane → X–Z) ===
        x_plane = np.linspace(-1, 1, 10)
        z_plane2 = np.linspace(-1, 1, 10)
        Xm, Zm = np.meshgrid(x_plane, z_plane2)
        Ym = np.full_like(Xm, np.mean(y))  # No audio input
        ax.plot_surface(Xm, Ym, Zm, color='red', alpha=0.2)
        ax.scatter(x[2], y[2], z[2], s=150, color='red', label='Mute Agent', edgecolor='k')
        ax.text(x[2], y[2], z[2]+0.03, "Mute", color='red', fontsize=10, weight='bold')

        # === 4️⃣ Distance lines between Normal–Blind & Normal–Mute ===
        ax.plot([x[0], x[1]], [y[0], y[1]], [z[0], z[1]], color='green', linestyle='--', alpha=0.6)
        ax.plot([x[0], x[2]], [y[0], y[2]], [z[0], z[2]], color='red', linestyle='--', alpha=0.6)

        # Annotate distances on lines
        ax.text((x[0]+x[1])/2, (y[0]+y[1])/2, (z[0]+z[1])/2, f"{dist_blind:.2f}", color='green', fontsize=8)
        ax.text((x[0]+x[2])/2, (y[0]+y[2])/2, (z[0]+z[2])/2, f"{dist_mute:.2f}", color='red', fontsize=8)

        # === 5️⃣ Axis & title setup ===
        ax.set_xlabel("Image Reasoning (X-axis)")
        ax.set_ylabel("Audio Reasoning (Y-axis)")
        ax.set_zlabel("Text Reasoning (Z-axis)")
        ax.set_title("3D Reasoning Embedding Space for Multimodal Agents", fontsize=12, pad=20)
        ax.view_init(elev=25, azim=45)
        ax.legend()

        # --- Save plot ---
        plot_url = save_plot(fig)

        # --- Redirect to reasoning page ---
        redirect_url = url_for(
            "reasoning_page",
            plot=plot_url,
            dist_blind=dist_blind,
            dist_mute=dist_mute,
            closest=closest,
            **{
                "x": list(x),
                "y": list(y),
                "z": list(z)
            }
        )
        return jsonify({"redirect_url": redirect_url})

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        })


@app.route("/reasoning_page")
def reasoning_page():
    plot = request.args.get("plot")
    dist_blind = request.args.get("dist_blind", 0)
    dist_mute = request.args.get("dist_mute", 0)
    closest = request.args.get("closest")
    x = request.args.getlist("x", type=float)
    y = request.args.getlist("y", type=float)
    z = request.args.getlist("z", type=float)

    # ✅ Convert to floats
    try:
        dist_blind = float(dist_blind)
        dist_mute = float(dist_mute)
    except ValueError:
        dist_blind, dist_mute = 0.0, 0.0

    return render_template(
        "reasoning_visual.html",
        plot_url=plot,
        x=x, y=y, z=z,
        dist_blind=dist_blind,
        dist_mute=dist_mute,
        closest=closest
    )



if __name__ == "__main__":
    app.run(debug=True)
