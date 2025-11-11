uv run python train.py $1 0 30 & sleep 120; \
uv run python train.py $1 30 60 & sleep 60; \
uv run python train.py $1 60 90 & sleep 60; \
uv run python train.py $1 90 120 & sleep 60; \
uv run python train.py $1 120 160 & wait