uv run python train.py $1 0 80 & sleep 500; \
uv run python train.py $1 80 160 & wait

uv run python predict-last.py . 0 158