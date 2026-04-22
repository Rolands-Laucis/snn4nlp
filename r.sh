source ".venv/bin/activate"

python experiments/cast_embeddings.py --embeddings_path "input_data/word_embeddings/glove/glove.twitter.27B.50d.txt" --out_path="input_data/word_embeddings/glove/glove_50d.pkl"

python experiments/cast_pos_input.py
python experiments/cast_ner_input.py

# python experiments/pos_task.py --limit 100 --input_mode "spatial" --epochs 10
# python experiments/pos_task.py --limit 100 --input_mode "temporal"

# phase 0 - hyper parameter tuning
for sim_steps in 10 15 20 25 30; do
    echo "Running phase-0 pos_task with sim_steps=${sim_steps}"
    python experiments/pos_task.py --input_mode "temporal" --epochs 5 --beta 0.95 --sim_steps "${sim_steps}" --output_file_prefix "var-sim_steps"
done
for beta in 0.5 0.75 0.9 0.95 0.99; do
    echo "Running phase-0 pos_task with beta=${beta}"
    python experiments/pos_task.py --input_mode "temporal" --epochs 5 --beta "${beta}" --sim_steps 20 --output_file_prefix "var-beta"
done