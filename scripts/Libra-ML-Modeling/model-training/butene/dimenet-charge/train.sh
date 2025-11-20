uv run python train.py $1 0 40 & sleep 500; \
uv run python train.py $1 40 80 & sleep 120; \
uv run python train.py $1 80 120 & sleep 120; \
uv run python train.py $1 120 160 & wait