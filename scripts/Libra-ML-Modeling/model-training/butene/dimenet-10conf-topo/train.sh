uv run python train.py $1 0 20 & sleep 500; \
uv run python train.py $1 20 40 & sleep 120; \
uv run python train.py $1 40 60 & sleep 120; \
uv run python train.py $1 60 80 & sleep 120; \
uv run python train.py $1 80 100 & sleep 120; \
uv run python train.py $1 100 120 & sleep 120; \
uv run python train.py $1 120 140 & sleep 120; \
uv run python train.py $1 140 160 & wait

uv run python predict-last.py . 0 158