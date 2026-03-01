FROM mambaorg/micromamba:1.5.8

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

RUN micromamba install -y -n base -c conda-forge \
    python=3.11 \
    flask=3.0.3 \
    gunicorn=22.0.0 \
    numpy=1.26.4 \
    pandas=2.2.3 \
    scipy \
    scikit-learn \
    umap-learn \
    hdbscan \
    rdkit \
    curl && \
    micromamba run -n base pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    micromamba clean --all --yes

RUN chmod +x /app/latentvis/backend/start.sh

ENV PYTHONUNBUFFERED=1
ENV PORT=5001

EXPOSE 5001

CMD ["/bin/bash", "/app/latentvis/backend/start.sh"]
