import json
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "latentvis" / "frontend" / "public" / "data" / "bic_aic"
    out_path = data_dir / "avg_fmi_cspa_by_layer.json"

    layer_values: dict[str, dict[str, list[float]]] = {}
    model_values: dict[str, dict[str, list[float]]] = {}

    for path in sorted(data_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        layer = data.get("layer")
        model = data.get("model")
        if not layer:
            continue

        fmi = data.get("avg_fmi_vs_10conf")
        if fmi is None:
            fmi = data.get("avg_fmi_vs_noaug")
        cspa = data.get("avg_cspa_vs_10conf")
        if cspa is None:
            cspa = data.get("avg_cspa_vs_noaug")

        if fmi is None or cspa is None:
            continue

        bucket = layer_values.setdefault(layer, {"fmi": [], "cspa": []})
        bucket["fmi"].append(float(fmi))
        bucket["cspa"].append(float(cspa))

        if model:
            m_bucket = model_values.setdefault(model, {"fmi": [], "cspa": []})
            m_bucket["fmi"].append(float(fmi))
            m_bucket["cspa"].append(float(cspa))

    layers_out = []
    for layer, vals in sorted(layer_values.items()):
        fmi_vals = vals["fmi"]
        cspa_vals = vals["cspa"]
        if not fmi_vals or not cspa_vals:
            continue
        layers_out.append(
            {
                "layer": layer,
                "avg_fmi": sum(fmi_vals) / len(fmi_vals),
                "avg_cspa": sum(cspa_vals) / len(cspa_vals),
                "count": min(len(fmi_vals), len(cspa_vals)),
            }
        )

    models_out = []
    for model, vals in sorted(model_values.items()):
        fmi_vals = vals["fmi"]
        cspa_vals = vals["cspa"]
        if not fmi_vals or not cspa_vals:
            continue
        models_out.append(
            {
                "model": model,
                "avg_fmi": sum(fmi_vals) / len(fmi_vals),
                "avg_cspa": sum(cspa_vals) / len(cspa_vals),
                "count": min(len(fmi_vals), len(cspa_vals)),
            }
        )

    out = {"layers": layers_out, "models": models_out}
    out_path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
