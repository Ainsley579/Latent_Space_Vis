#!/usr/bin/env python3
from __future__ import annotations

from flask import Flask, jsonify, request


app = Flask(__name__)


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/generate-range", methods=["POST", "OPTIONS"])
def generate_range():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    min_value = payload.get("min")
    max_value = payload.get("max")

    print(f"[flask-backend] generate-range min={min_value}, max={max_value}", flush=True)
    return jsonify({"ok": True, "min": min_value, "max": max_value})


@app.route("/search-smiles", methods=["POST", "OPTIONS"])
def search_smiles():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    smiles = payload.get("smiles")
    print(f"[flask-backend] search-smiles smiles={smiles}", flush=True)
    return jsonify({"ok": True, "smiles": smiles})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
