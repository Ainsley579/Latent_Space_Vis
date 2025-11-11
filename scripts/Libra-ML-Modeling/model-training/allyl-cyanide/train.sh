uv run python train.py $1 0 15 & sleep 500; \
uv run python train.py $1 15 30 & sleep 120; \
uv run python train.py $1 30 60 & wait