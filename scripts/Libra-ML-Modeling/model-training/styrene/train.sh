uv run python train.py $1 0 30 & sleep 60; \
uv run python train.py $1 30 60 & sleep 30; \
uv run python train.py $1 60 90 & wait