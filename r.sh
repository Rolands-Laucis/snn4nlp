source ".venv/bin/activate"

# python experiments/cast_embeddings.py --embeddings_path "input_data/word_embeddings/glove/glove.twitter.27B.50d.txt" --out_path="input_data/word_embeddings/glove/glove_50d.pkl"

# python experiments/cast_pos_input.py --min_sentence_length 5 --max_sentence_length 40
# python experiments/cast_ner_input.py

# python experiments/E0.py --limit 100 --input_mode "spatial" --epochs 10
# python experiments/E0.py --limit 100 --input_mode "temporal"

# phase 0 - hyper parameter tuning
# for sim_steps in 10 15 20 25 30; do
#     echo "Running phase-0 E0 with sim_steps=${sim_steps}"
#     python experiments/E0.py --input_mode "temporal" --epochs 5 --beta 0.95 --sim_steps "${sim_steps}" --output_file_prefix "var-sim_steps" --limit 100
# done
# for beta in 0.5 0.75 0.9 0.95 0.99; do
#     echo "Running phase-0 E0 with beta=${beta}"
#     python experiments/E0.py --input_mode "temporal" --epochs 5 --beta "${beta}" --sim_steps 20 --output_file_prefix "var-beta" --limit 100
# done

# the best found were sim_steps=20 and beta=0.95

# phase 1
# E1-A
# python experiments/E0.py 
#     / --input_mode "temporal" 
#     / --epochs 20 
#     / --beta 0.95 
#     / --sim_steps 20 
#     / --batch_size 16
#     / --encoding_method "poisson" 
#     / --decoding_method "spike_count" 
#     / --output_file_prefix "pois_sc" 
# # E1-B
# python experiments/E1.py
#     / --input_mode "temporal" 
#     / --epochs 20 
#     / --beta 0.95 
#     / --sim_steps 20 
#     / --batch_size 16
#     / --encoding_method "latency" 
#     / --decoding_method "ttfs" 
#     / --output_file_prefix "lat_ttfs" 
# E1-C
# python experiments/E1.py --input_mode "temporal" --epochs 20 --beta 0.95 --sim_steps 20 --encoding_method "direct" --decoding_method "spike_count" --output_file_prefix "dir_sc" --batch_size 16

# python -u experiments/E1.py --input_mode "temporal" --epochs 50 --batch_size 8 --beta 0.95 --sim_steps 10 --encoding_method "latency" --decoding_method "ttfs" --output_file_prefix "E1_ttfs_10" --limit 70000
python -u experiments/E1.py --input_mode "spatial" --epochs 50 --batch_size 8 --beta 0.9 --sim_steps 40 --encoding_method "poisson" --decoding_method "spike_count" --output_file_prefix "E1_poisson" --limit 70000
python -u experiments/E1.py --input_mode "temporal" --epochs 50 --batch_size 8 --beta 0.9 --sim_steps 40 --encoding_method "latency" --decoding_method "ttfs" --output_file_prefix "E1_ttfs_40" --limit 70000
# python -u experiments/E1.py --input_mode "temporal" --epochs 50 --batch_size 16 --beta 0.95 --sim_steps 10 --neuron_model "synaptic" --output_file_prefix "E1_synaptic_10" 
# python -u experiments/E1.py --input_mode "temporal" --epochs 50 --batch_size 16 --beta 0.95 --sim_steps 20 --neuron_model "synaptic" --output_file_prefix "E1_synaptic_20" 
# python experiments/E1.py --input_mode "temporal" --epochs 50 --batch_size 16 --beta 0.95 --sim_steps 10 --neuron_model "qlif" --output_file_prefix "E1_qlif"