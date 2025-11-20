uv run chemprop predict \
        --model-paths models/ \
        --test-path data/experimental/data/experimental.csv \
        --descriptors-path data/experimental/data/reaction_conditions.npz \
        --preds-path data/experimental/results/preds.csv 