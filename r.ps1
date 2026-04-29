.\venv\Scripts\Activate.ps1

# ---DATA PREP---
# python experiments/cast_embeddings.py --embeddings_path "input_data\word_embeddings\glove\glove.twitter.27B.50d.txt" --out_path="input_data\word_embeddings\glove\glove_50d.pkl" --normalization_mode "tanh"
# python experiments/cast_embeddings.py --embeddings_path "input_data\word_embeddings\glove\glove.twitter.27B.50d.txt" --out_path="input_data\word_embeddings\glove\glove_50d.pkl" --normalization_mode "sigmoid"
#rescale and L2 norm were giving unintuitive results with bad training performance, where the models didnt seem to learn at all
# sigmoid is *4 because it makes the sigmoid steeper, pushing more values closer to 0 or 1. This seems intuitive to use up more of the bulk values in the desired range
# sigmoid seems to perform the same as tanh, but might be faster to compute. I stick to tanh

# python experiments/cast_sent_input.py --min_sentence_length 5 --max_sentence_length 10
# python experiments/cast_pos_input.py --min_sentence_length 5
# python experiments/cast_ner_input.py


# tests
# python experiments/readers.py
# python experiments/snn_util.py


# ---EXPERIMENTS---

$lr = 1e-4

# phase 0 - hyper parameter tuning for sentiment
# foreach ($sim_steps in @(15, 20, 25, 30, 40)) {
#     foreach ($beta in @(0.5, 0.75, 0.9, 0.95, 0.99)) {
#         Write-Host "Running phase-0-A with sim_steps=$sim_steps beta=$beta"

#         python experiments/E_sent.py --input_mode "spatial" --encoding_method "poisson" --decoding_method "spike_count" --epochs 10 --beta $beta --sim_steps $sim_steps --threshold 1 --threshold_layer_scalars "[1, 0.8, 0.7]" --limit 1000 --learning_rate $lr --batch_size 64 --output_file_prefix "hypr-1"
#     }
# }
# best for sentiment with spatial input found to be sim_steps=40 (will use 25) and beta=0.9
$sim_steps = 25 #a bit less to save on compute for temporal input
$beta = 0.9

# foreach ($l1 in @(0.6, 0.7, 0.8, 0.9)) {
#     foreach ($l2 in @(1, 1.1, 1.2, 1.5, 2)) { #$l2 /= $l1
#         $l2 = [math]::Round($l1/$l2, 2) # round to 2 decimal places
#         Write-Host "Running phase-0-B with l1=$l1 l2=$l2"

#         python experiments/E_sent.py --input_mode "spatial" --encoding_method "poisson" --decoding_method "spike_count" --epochs 10 --beta $beta --sim_steps $sim_steps --threshold 1 --threshold_layer_scalars "[1, $l1, $l2]" --limit 1000 --learning_rate $lr --batch_size 64 --output_file_prefix "hypr-2"
#     }
# }
# best for sentiment with spatial input found
$threshold_layer_scalars = "[1, 0.8, 0.7]" #0.8 and 1.2, but most values are close

# consistent settings for all runs
$limit = 20000
$batch_size = 32
$epochs = 50

# tests
# python experiments/E_sent.py --input_mode "spatial" --encoding_method "latency" --decoding_method "spike_count" --save --epochs 10 --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --output_file_prefix "sigmoid"


# sent full run
# python experiments/E_sent.py --input_mode "spatial" --encoding_method "poisson" --decoding_method "spike_count" --save --eval --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size 
# python experiments/E_sent.py --input_mode "temporal" --encoding_method "poisson" --decoding_method "spike_count" --save --eval --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size
# temporal mode is quite less accurate in train, but the test score is very close to train, meaning it prob generalizes better than spatial mode.
# need to test, if other hyper params would improve it:

# foreach ($sim_steps in @(15, 20, 25, 30, 40)) {
#     Write-Host "Running phase-1-B with sim_steps=$sim_steps beta=$beta"

#     python experiments/E_sent.py --input_mode "temporal" --encoding_method "poisson" --decoding_method "spike_count" --epochs 10 --beta $beta --learn_beta True --sim_steps $sim_steps --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --limit 1000 --learning_rate $lr --batch_size 64 --output_file_prefix "hypr-3"
# }
# the best accuracy is still with sim_steps=25, and learning beta doesn't seem to help much, as the learned beta values are all close to the initial value of 0.9, so will keep it fixed at 0.9 for all runs.
# for the sake of computational cost, will use spatial input for the rest of the experiments, as it trains much faster, and has higher train accuracy


# spatial wins; vary encoding and decoding method
# python experiments/E_sent.py --input_mode "spatial" --encoding_method "latency" --decoding_method "spike_count" --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --save --eval --output_file_prefix "lat_sc"
# python experiments/E_sent.py --input_mode "spatial" --encoding_method "latency" --decoding_method "ttfs" --ttfs_temporal_loss "ce_temporal_loss" --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --save --eval --output_file_prefix "lat_ttfs_ce"
# python experiments/E_sent.py --input_mode "spatial" --encoding_method "latency" --decoding_method "ttfs" --ttfs_temporal_loss "mse_temporal_loss" --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --save --eval --output_file_prefix "lat_ttfs_mse"
# TODO FULL temporal setup, just to double check:
# python experiments/E_sent.py --input_mode "temporal" --encoding_method "latency" --decoding_method "ttfs" --ttfs_temporal_loss "ce_temporal_loss" --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --save --eval --output_file_prefix "temporal_lat_ttfs_ce"



# python experiments/E_pos.py --input_mode "temporal" --epochs 50 --beta 0.95 --sim_steps 20 --limit 1000 --encoding_method "latency" --decoding_method "ttfs" --output_file_prefix "tmp_ttfs" 
# python experiments/E_pos.py --input_mode "temporal" --epochs 50 --beta 0.95 --sim_steps 50 --limit 1000 --encoding_method "latency" --ttfs_temporal_loss "ce_temporal_loss" --decoding_method "ttfs" --output_file_prefix "tmp_ttfs" 
# python experiments/E_pos.py --input_mode "temporal" --epochs 50 --beta 0.5 --sim_steps 50 --limit 1000 --encoding_method "latency" --ttfs_temporal_loss "mse_temporal_loss" --decoding_method "ttfs" --output_file_prefix "tmp_ttfs" 
# python experiments/E_pos.py --input_mode "spatial" --epochs 50 --beta 0.95 --sim_steps 10 --limit 1000 --encoding_method "latency" --decoding_method "ttfs" --output_file_prefix "tmp_spatial_ttfs" 
# python experiments/E_pos.py --input_mode "spatial" --epochs 50 --beta 0.9 --sim_steps 40 --limit 1000 --encoding_method "poisson" --decoding_method "spike_count" --output_file_prefix "tmp_pois" 

# python experiments/E_pos.py --input_mode "spatial" --epochs 50 --beta 0.95 --sim_steps 10 --encoding_method "poisson" --decoding_method "spike_count" --output_file_prefix "E_pos_poisson" 
# python experiments/E_pos.py --input_mode "temporal" --epochs 50 --beta 0.95 --sim_steps 10 --encoding_method "latency" --decoding_method "ttfs" --output_file_prefix "E_pos_ttfs_10" 
# python experiments/E_pos.py --input_mode "temporal" --epochs 50 --beta 0.95 --sim_steps 20 --encoding_method "latency" --decoding_method "ttfs" --output_file_prefix "E_pos_ttfs_20" 
# python experiments/E_pos.py --input_mode "temporal" --epochs 50 --beta 0.95 --sim_steps 10 --neuron_model "synaptic" --output_file_prefix "E_pos_synaptic_10" 
# python experiments/E_pos.py --input_mode "temporal" --epochs 50 --beta 0.95 --sim_steps 20 --neuron_model "synaptic" --output_file_prefix "E_pos_synaptic_20" 
# python experiments/E_pos.py --input_mode "temporal" --epochs 50 --beta 0.95 --sim_steps 10 --neuron_model "qlif" --output_file_prefix "E_pos_qlif"