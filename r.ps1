.\venv\Scripts\Activate.ps1

# python experiments/cast_embeddings.py --embeddings_path "input_data\word_embeddings\glove\glove.twitter.27B.50d.txt" --out_path="input_data\word_embeddings\glove\glove_50d.pkl"

# python experiments/cast_pos_input.py --min_sentence_length 5 --max_sentence_length 40
# python experiments/cast_ner_input.py

# python experiments/E0.py --limit 100 --input_mode "spatial" --epochs 10
# python experiments/E0.py --limit 100 --input_mode "temporal"

# phase 0 - hyper parameter tuning
# foreach ($sim_steps in @(10, 15, 20, 25, 30)) {
#     Write-Host "Running phase-0 E0 with sim_steps=$sim_steps"
#     python experiments/E0.py --input_mode "temporal" --epochs 5 --beta 0.95 --sim_steps $sim_steps --output_file_prefix "var-sim_steps" --limit 10
# }
# foreach ($beta in @(0.5, 0.75, 0.9, 0.95, 0.99)) {
#     Write-Host "Running phase-0 E0 with beta=$beta"
#     python experiments/E0.py --input_mode "temporal" --epochs 5 --beta $beta --sim_steps 20 --output_file_prefix "var-beta"
# }

# python experiments/E1.py --input_mode "temporal" --epochs 5 --beta 0.95 --sim_steps 20 --encoding "poisson" --decoding "spike_count" --output_file_prefix "E1_poisson" --limit 100
# python experiments/E1.py --input_mode "temporal" --epochs 5 --beta 0.95 --sim_steps 20 --encoding "latency" --decoding "ttfs" --output_file_prefix "E1_ttfs" --limit 100
# python experiments/E1.py --input_mode "temporal" --epochs 5 --beta 0.95 --sim_steps 20 --neuron_model "synaptic" --output_file_prefix "E1_synaptic" --limit 100
# python experiments/E1.py --input_mode "temporal" --epochs 5 --beta 0.95 --sim_steps 20 --neuron_model "qlif" --output_file_prefix "E1_qlif" --limit 100