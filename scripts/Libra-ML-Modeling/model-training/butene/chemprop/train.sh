for split in $(seq "$1" "$2")
do
    echo "Training model for split ${split}..."
    uv run chemprop train \
        --data-path data/splits-csv/combined-${split}.csv \
        --task-type regression \
        --loss-function mse \
        --output-dir models/split_${split}/ \
        --descriptors-path data/splits-npz/reaction_conditions-${split}.npz \
        --ensemble-size 1 \
        --splits-column split

    echo "Predicting for split ${split}..."
    mkdir -p results/split_${split}
    uv run chemprop predict \
        --model-paths models/split_${split}/ \
        --test-path data/splits-csv/combined-${split}.csv \
        --descriptors-path data/splits-npz/reaction_conditions-${split}.npz \
        --preds-path results/split_${split}/preds.csv 
done