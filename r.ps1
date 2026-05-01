.\venv\Scripts\Activate.ps1

# ---DATA PREP---
# python experiments/cast_embeddings.py --embeddings_path "input_data\word_embeddings\glove\glove.twitter.27B.50d.txt" --out_path="input_data\word_embeddings\glove\glove_50d_sigmoid.pkl" --normalization_mode "sigmoid"
# python experiments/cast_embeddings.py --embeddings_path "input_data\word_embeddings\glove\glove.twitter.27B.25d.txt" --out_path="input_data\word_embeddings\glove\glove_25d_sigmoid.pkl" --normalization_mode "sigmoid"
# python experiments/cast_embeddings.py --embeddings_path "input_data\word_embeddings\glove\glove.twitter.27B.100d.txt" --out_path="input_data\word_embeddings\glove\glove_100d_sigmoid.pkl" --normalization_mode "sigmoid"
#rescale and L2 norm were giving unintuitive results with bad training performance, where the models didnt seem to learn at all
# sigmoid is *4 because it makes the sigmoid steeper, pushing more values closer to 0 or 1. This seems intuitive to use up more of the bulk values in the desired range
# sigmoid seems to perform the same as tanh, but might be faster to compute. So i use sigmoid

# python experiments/cast_sent_input.py --min_sentence_length 5 --max_sentence_length 10 --embeddings_path "input_data\word_embeddings\glove\glove_50d_sigmoid.pkl"
# python experiments/cast_sent_input.py --min_sentence_length 5 --max_sentence_length 10 --embeddings_path "input_data\word_embeddings\glove\glove_25d_sigmoid.pkl"
# python experiments/cast_sent_input.py --min_sentence_length 5 --max_sentence_length 10 --embeddings_path "input_data\word_embeddings\glove\glove_100d_sigmoid.pkl"

# python experiments/cast_pos_input.py --min_sentence_length 5
# python experiments/cast_ner_input.py


# tests
# python experiments/readers.py
# python experiments/snn_util.py


# ---EXPERIMENTS---
# ---PHASE 0 - HYPERPARAMETER TUNING---

$lr = 1e-4 #SNNLP paper used 1e-5, which is modest imo. Seems like learning is quite stable, and can handle higher learning rates, so will use 1e-4 to speed up training.

# phase 0 - hyper parameter tuning for sentiment
# foreach ($sim_steps in @(15, 20, 25, 30, 40)) {
#     foreach ($beta in @(0.5, 0.75, 0.9, 0.95, 0.99)) {
#         Write-Host "Running phase-0-A with sim_steps=$sim_steps beta=$beta"

#         python experiments/E_sent.py --input_mode "spatial" --encoding_method "poisson" --decoding_method "spike_count" --epochs 10 --beta $beta --sim_steps $sim_steps --threshold 1 --threshold_layer_scalars "[1, 0.8, 0.7]" --limit 1000 --learning_rate $lr --batch_size 64 --output_file_prefix "hypr-1"
#     }
# }
# best for sentiment with spatial input found to be sim_steps=40 (will use 25) and beta=0.95
$sim_steps = 30 #a bit less to save on compute for temporal input
$beta = 0.95

# foreach ($l1 in @(0.6, 0.7, 0.8, 0.9)) {
#     foreach ($l2 in @(1, 1.1, 1.2, 1.5, 2)) { #$l2 /= $l1
#         $l2 = [math]::Round($l1/$l2, 2) # round to 2 decimal places
#         Write-Host "Running phase-0-B with l1=$l1 l2=$l2"

#         python experiments/E_sent.py --input_mode "spatial" --encoding_method "poisson" --decoding_method "spike_count" --epochs 10 --beta $beta --sim_steps $sim_steps --threshold 1 --threshold_layer_scalars "[1, $l1, $l2]" --limit 1000 --learning_rate $lr --batch_size 64 --output_file_prefix "hypr-2"
#     }
# }
# best for sentiment with spatial input found
$threshold_layer_scalars = "[1, 0.7, 0.7]"

# consistent settings for all runs
$limit = 20000
$batch_size = 32
$epochs = 50

# tests
# python experiments/E_sent.py --input_mode "spatial" --encoding_method "latency" --decoding_method "spike_count" --save --epochs 10 --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --output_file_prefix "sigmoid"


# ---PHASE 1 - INPUT MODE---
# python experiments/E_sent.py --input_mode "spatial" --encoding_method "poisson" --decoding_method "spike_count" --save --eval --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size 
# python experiments/E_sent.py --input_mode "temporal" --encoding_method "poisson" --decoding_method "spike_count" --save --eval --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size
# temporal mode is quite less accurate in train, but the test score is very close to train for both tanh and sigmoid, meaning it prob generalizes better than spatial mode.
# need to test, if other hyper params would improve it:

# TODO on sigmoid:
# foreach ($sim_steps in @(15, 20, 25, 30, 40)) {
#     Write-Host "Running phase-1-B with sim_steps=$sim_steps beta=$beta"

#     python experiments/E_sent.py --input_mode "temporal" --encoding_method "poisson" --decoding_method "spike_count" --epochs 10 --beta $beta --learn_beta True --sim_steps $sim_steps --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --limit 1000 --learning_rate $lr --batch_size 64 --output_file_prefix "hypr-3"
# }
# the best accuracy is still with sim_steps=25, and learning beta doesn't seem to help much, as the learned beta values are all close to the initial value of 0.9, so will keep it fixed at 0.9 for all runs.
# for the sake of computational cost, will use spatial input for the rest of the experiments, as it trains much faster, and has higher train accuracy


# ---PHASE 1 - ENCODING AND DECODING---
# spatial wins; vary encoding and decoding method
# python experiments/E_sent.py --input_mode "spatial" --encoding_method "latency" --decoding_method "spike_count" --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --save --eval --output_file_prefix "lat_sc"
# python experiments/E_sent.py --input_mode "spatial" --encoding_method "latency" --decoding_method "ttfs" --ttfs_temporal_loss "ce_temporal_loss" --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --save --eval --output_file_prefix "lat_ttfs_ce"
# python experiments/E_sent.py --input_mode "spatial" --encoding_method "latency" --decoding_method "ttfs" --ttfs_temporal_loss "mse_temporal_loss" --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --save --eval --output_file_prefix "lat_ttfs_mse"
# TODO FULL temporal setup, just to double check:
# python experiments/E_sent.py --input_mode "temporal" --encoding_method "latency" --decoding_method "ttfs" --ttfs_temporal_loss "ce_temporal_loss" --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --save --eval --output_file_prefix "temporal_lat_ttfs_ce"

# ---PHASE 2 - other confounding variables - glove emb dim, neuron model---
# python experiments/E_sent.py --input_file_prefix "sent_d25" --input_mode "spatial" --encoding_method "latency" --decoding_method "spike_count" --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --save --eval --output_file_prefix "lat_sc_25"
# python experiments/E_sent.py --input_file_prefix "sent_d100" --input_mode "spatial" --encoding_method "latency" --decoding_method "spike_count" --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --save --eval --output_file_prefix "lat_sc_100"


# ---PHASE 3 - NLP tasks---
# python experiments/E_pos.py --input_mode "spatial" --epochs 50 --beta 0.9 --sim_steps 40 --limit 1000 --encoding_method "poisson" --decoding_method "spike_count" --output_file_prefix "tmp_pois"
# for POS also test spatial vs temporal input with shuffled token order in eval on trained models to see if either degrades and by how much, which would indicate whether temporal actually inputs token order implicitly
