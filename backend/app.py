from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app)  # allow requests from Next.js (localhost:3000)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    data_size = data.get("dataSize")
    feature_count = data.get("featureCount")
    model = data.get("model")
    window_id = data.get("windowId")

    print("Generate requested:")
    print("  dataSize     =", data_size)
    print("  featureCount =", feature_count)
    print("  model        =", model)
    print("  windowId     =", window_id)

    # Run the python tracking script
    output_base = "../../frontend/public/data"
    window_output_dir = os.path.join(output_base, str(window_id))
    os.makedirs(window_output_dir, exist_ok=True)
    cmd = [
        "python",
        "../scripts/GNNGMMAnalyzer/generate_real_id_tracking_csv.py",
        "--embeddings_dir", f"../scripts/dimenet{model}_embeddings",
        "--output_dir", window_output_dir,
    ]

    try:
        subprocess.run(cmd, check=True)
        # Run the scatter plot data script
        scatter_cmd = [
            "python",
            "../scripts/GNNGMMAnalyzer/generate_scatter_plot_data.py",
            "--embeddings_dir", f"../scripts/dimenet{model}_embeddings",
            "--output_dir", os.path.join(window_output_dir, "scatter_plot_data_by_layer")
        ]
        subprocess.run(scatter_cmd, check=True)
        return jsonify({
            "status": "ok",
            "windowId": window_id,
            "layer_csv": f"/data/{window_id}/real_id_layer_transitions/real_id_layer_transitions.csv",
            "scatter_csv": f"/data/{window_id}/scatter_plot_data_by_layer"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/")
def home():
    return "Backend running!"


if __name__ == "__main__":
    app.run(port=5001, debug=True)