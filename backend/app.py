from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GNNGMM_DIR = PROJECT_ROOT / "scripts" / "GNNGMMAnalyzer"
sys.path.append(str(GNNGMM_DIR))

from embedding_pipeline_core import load_embeddings_from_files
from embedding_pipeline import EmbeddingPipeline

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
    layer_unique = ""
    if data_size==374:
        layer_unique = "_unique"
    elif data_size==3487:
        layer_unique = "_all"
    cmd = [
        "python",
        "../scripts/GNNGMMAnalyzer/generate_real_id_tracking_csv.py",
        "--embeddings_dir", f"../scripts/dimenet{model}_embeddings"+layer_unique,
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
    

@app.route("/anyembedding", methods=["POST"])
def anyembedding():
    """
    Backend version of run_embedding_pipeline.py
    - Same input semantics as CLI
    - Returns scatter + sankey directly
    """
    data = request.json or {}

    embeddings_dir = data.get("embeddings_dir")
    layer_files = data.get("layer_files")
    layer_names = data.get("layer_names")

    n_components = data.get("n_components")
    id_mapping_file = data.get("id_mapping")
    train_data_file = data.get("train_data")
    test_data_file = data.get("test_data")

    if embeddings_dir is None:
        return jsonify({
            "status": "error",
            "message": "embeddings_dir is required"
        }), 400

    try:
        embeddings_dir = Path(embeddings_dir)

        if layer_files and layer_names:
            layer_files_dict = dict(zip(layer_names, layer_files))
            embeddings = load_embeddings_from_files(
                embeddings_dir,
                layer_files_dict
            )
        else:
            embeddings = load_embeddings_from_files(embeddings_dir)

        id_mapping = None
        if id_mapping_file:
            with open(id_mapping_file, "r") as f:
                raw = json.load(f)
            id_mapping = {int(k): int(v) for k, v in raw.items()}

        metadata = {}
        if train_data_file:
            with open(train_data_file, "r") as f:
                train_data = json.load(f)
            metadata["train_data_size"] = len(train_data)

        if test_data_file:
            with open(test_data_file, "r") as f:
                test_data = json.load(f)
            metadata["test_data_size"] = len(test_data)
            
        tmp_output_dir = Path("../frontend/public/data/tmp")
        tmp_output_dir.mkdir(parents=True, exist_ok=True)

        pipeline = EmbeddingPipeline(
            embeddings=embeddings,
            output_dir=tmp_output_dir,         
            id_mapping=id_mapping,
            n_components=n_components,
            metadata=metadata
        )

        pipeline.fit_gmms()
        pipeline.compute_mds()
        # pipeline.run()

        scatter_df = pipeline.generate_scatter_plot_data()
        sankey_df = pipeline.generate_sankey_data()

        n_layers = len(embeddings)
        first_layer = next(iter(embeddings.values()))
        n_samples = first_layer.shape[0]

        return jsonify({
            "status": "ok",
            "scatter": scatter_df.to_dict(orient="records"),
            "sankey": sankey_df.to_dict(orient="records"),
            "metadata": {
                "n_samples": n_samples,
                "n_layers": n_layers,
                **metadata
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/")
def home():
    return "Backend running!"


if __name__ == "__main__":
    app.run(port=5001, debug=True)