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

# python experiments/cast_pos_input.py --min_sentence_length 5 --max_sentence_length 20 --train_test_split_ratio 0.9 --embeddings_path "input_data\word_embeddings\glove\glove_100d_sigmoid.pkl"
# min 5 because the window size is 5, so anything less than that would be weird to train on. Max 20 because the dataset has a max 157 and that is far too big of an input to the network.

# python experiments/cast_ner_input.py

# tests
# python experiments/readers.py
# python experiments/snn_util.py


# ---EXPERIMENTS---
# ---PHASE 0 - HYPERPARAMETER TUNING---

$lr = 1e-4 #SNNLP paper used 1e-5, which is modest imo. Seems like learning is quite stable, and can handle higher learning rates, so will use 1e-4 to speed up training.

# hype-1 parameter tuning for sentiment
# foreach ($sim_steps in @(15, 20, 25, 30, 40)) {
#     foreach ($beta in @(0.5, 0.75, 0.9, 0.95, 0.99)) {
#         Write-Host "Running phase-0-A with sim_steps=$sim_steps beta=$beta"

#         python experiments/E_sent.py --input_mode "spatial" --encoding_method "poisson" --decoding_method "spike_count" --epochs 10 --beta $beta --sim_steps $sim_steps --threshold 1 --threshold_layer_scalars "[1, 0.8, 0.7]" --limit 1000 --learning_rate $lr --batch_size 64 --output_file_prefix "hypr-1"
#     }
# }
# best for sentiment with spatial input found to be sim_steps=40 and beta=0.95
$sim_steps = 30 #a bit less to save on compute for temporal input
$beta = 0.95

# hyper-2
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
$limit = 20000 #for sent this is higher than the number of samples in the dataset, but for POS it is lower.
$batch_size = 32
$epochs = 50

# tests
# python experiments/E_sent.py --input_mode "spatial" --encoding_method "poisson" --decoding_method "spike_count" --epochs 1 --beta $beta --learn_beta True --per_neuron_params True --sim_steps $sim_steps --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --limit 1000 --learning_rate $lr --batch_size 64 --output_file_prefix "hypr-3"


# ---PHASE 1 - INPUT MODE---
# python experiments/E_sent.py --input_mode "spatial" --encoding_method "poisson" --decoding_method "spike_count" --save --eval --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size 
# python experiments/E_sent.py --input_mode "temporal" --encoding_method "poisson" --decoding_method "spike_count" --save --eval --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size
# temporal mode is quite less accurate in train, but the test score is very close to train for both tanh and sigmoid, meaning it prob generalizes better than spatial mode.
# need to test, if other hyper params would improve it:

# hyper-3 for temporal mode
# foreach ($sim_steps in @(15, 20, 25, 30, 40)) {
#     Write-Host "Running phase-1-B with sim_steps=$sim_steps beta=$beta"

#     python experiments/E_sent.py --input_mode "temporal" --encoding_method "poisson" --decoding_method "spike_count" --epochs 10 --beta $beta --learn_beta True --sim_steps $sim_steps --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --limit 1000 --learning_rate $lr --batch_size 64 --output_file_prefix "hypr-3"
# }
# the best accuracy is with sim_steps=20 and 30, but not 25, and learning beta at 0.95, which means it probably tries to keep all past context right to the end.
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

# ANN equivalent for dimension comparison.
# python experiments/E_sent_ann-mlp.py --input_file_prefix "sent_d50" --output_file_prefix "ann_mlp_50d" --limit $limit --learning_rate $lr --batch_size $batch_size --epochs $epochs --save --eval
# python experiments/E_sent_ann-mlp.py --input_file_prefix "sent_d50" --output_file_prefix "ann_mlp_50d" --limit 1000 --learning_rate $lr --batch_size $batch_size --epochs 1 --save --eval
# python experiments/E_sent_ann-mlp.py --input_file_prefix "sent_d25" --output_file_prefix "ann_mlp_25d" --limit $limit --learning_rate $lr --batch_size $batch_size --epochs $epochs --save --eval
# python experiments/E_sent_ann-mlp.py --input_file_prefix "sent_d100" --output_file_prefix "ann_mlp_100d" --limit $limit --learning_rate $lr --batch_size $batch_size --epochs $epochs --save --eval

# python experiments/E_sent_ann-lstm.py --input_file_prefix "sent_d25" --output_file_prefix "ann_lstm_25" --limit $limit --learning_rate $lr --batch_size $batch_size --epochs $epochs --save --eval
# python experiments/E_sent_ann-lstm.py --input_file_prefix "sent_d50" --output_file_prefix "ann_lstm_50" --limit $limit --learning_rate $lr --batch_size $batch_size --epochs $epochs --save --eval
# python experiments/E_sent_ann-lstm.py --input_file_prefix "sent_d100" --output_file_prefix "ann_lstm_100" --lstm_bidirectional False --limit $limit --learning_rate $lr --batch_size $batch_size --epochs $epochs --save --eval
# python experiments/E_sent_ann-lstm.py --input_file_prefix "sent_d100" --output_file_prefix "ann_bilstm_100" --lstm_bidirectional True --limit $limit --learning_rate $lr --batch_size $batch_size --epochs $epochs --save --eval


# neuron model
# python experiments/E_sent.py --diagnose --input_file_prefix "sent_d25" --input_mode "temporal" --encoding_method "poisson" --decoding_method "spike_count" --neuron_model "synaptic" --output_file_prefix "lat_sc_100_synaptic" --per_neuron_params True --learn_alpha True --epochs 1 --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit 1000 --learning_rate $lr --batch_size 64
# python experiments/E_sent.py --input_file_prefix "sent_d100" --neuron_model "qlif" --input_mode "spatial" --encoding_method "latency" --decoding_method "spike_count" --output_file_prefix "lat_sc_100_qlif" --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --save --eval
# python experiments/E_sent.py --input_file_prefix "sent_d100" --neuron_model "synaptic" --input_mode "spatial" --encoding_method "latency" --decoding_method "spike_count" --output_file_prefix "lat_sc_100_synaptic" --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --save --eval
# synaptic does really well and achieves 99% train acc in the first 20 epochs with learn alpha false
# qlif is slightly worse than lif
# python experiments/E_sent.py --input_file_prefix "sent_d100" --neuron_model "synaptic" --learn_alpha True --input_mode "spatial" --encoding_method "latency" --decoding_method "spike_count" --output_file_prefix "lat_sc_100_synaptic_learn" --epochs $epochs --beta $beta --threshold 1 --threshold_layer_scalars $threshold_layer_scalars --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size --save --eval


# python experiments/E_sent_eval.py --input_file_prefix "sent_d100" --model_path "output_results\E_sent\main-2\lat_sc_100_2026-04-30_11-03-04_e-50_s-30_spatial.pt" --encoding_method "latency" --decoding_method "spike_count" --sim_steps $sim_steps --batch_size $batch_size --output_json "./100d_eval.json"
# python experiments/E_sent_eval.py --input_file_prefix "sent_d100" --model_path "output_results\E_sent\neuron_model\lat_sc_100_lif_2026-05-12_07-55-49_e-50_s-30_spatial.pt" --encoding_method "latency" --decoding_method "spike_count" --sim_steps $sim_steps --batch_size $batch_size --output_json "./100d_eval.json"
# python experiments/E_sent_eval.py --input_file_prefix "sent_d50" --model_path "output_results\E_sent\main\lat_sc_2026-04-30_08-23-34_e-50_s-30_spatial.json" --encoding_method "latency" --decoding_method "spike_count" --sim_steps $sim_steps --batch_size $batch_size --output_json "./50d_eval.json"
# python experiments/E_sent_eval.py --input_file_prefix "sent_d25" --model_path "output_results\E_sent\main-2\lat_sc_25_2026-05-01_07-39-56_e-50_s-30_spatial.pt" --encoding_method "latency" --decoding_method "spike_count" --sim_steps $sim_steps --batch_size $batch_size --output_json "./25d_eval.json"


# ---PHASE 3 - UPOS task---
$threshold_layer_scalars = "[1, 1, 1]"

# hyper params
# foreach ($sim_steps in @(15, 20, 25, 30, 40)) {
#     foreach ($beta in @(0.8, 0.9, 0.95, 0.99)) {
#         python experiments/E_pos.py --input_mode "spatial" --encoding_method "latency" --output_file_prefix "hypr-1/upos_hypr-1" --epochs 5 --beta $beta --sim_steps $sim_steps --limit 1000 --learning_rate $lr --batch_size 64 --threshold_layer_scalars $threshold_layer_scalars
#     }
# }
$sim_steps = 20
$beta = 0.9

# hypr-2 for alpha
# foreach ($alpha in @(0.8, 0.9, 0.95, 0.99)) {
#    python experiments/E_pos.py --input_mode "spatial" --encoding_method "latency" --output_file_prefix "hypr-2/upos_hypr-2" --epochs 5 --beta $beta --alpha $alpha --sim_steps $sim_steps --limit 1000 --learning_rate $lr --batch_size 64 --threshold_layer_scalars $threshold_layer_scalars
# }
$alpha = 0.95

# MODELS
# python experiments/E_pos.py --diagnose --input_mode "spatial" --encoding_method "latency" --output_file_prefix "diag" --epochs 1 --beta $beta --sim_steps $sim_steps --limit 20000 --learning_rate $lr --batch_size 64
# python experiments/E_pos.py --input_mode "spatial" --encoding_method "latency" --output_file_prefix "tmp" --epochs 1 --beta $beta --alpha $alpha --sim_steps $sim_steps --limit 1000 --learning_rate $lr --batch_size 32
# python experiments/E_pos.py --save --eval --input_mode "spatial" --encoding_method "latency" --output_file_prefix "upos-win-snn" --epochs $epochs --beta $beta --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size
# python experiments/E_pos.py --save --eval --input_mode "temporal" --encoding_method "latency" --output_file_prefix "upos-win-snn" --epochs $epochs --beta $beta --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size

# for POS also test spatial vs temporal input with shuffled token order in eval on trained models to see if either degrades and by how much, which would indicate whether temporal actually inputs token order implicitly
# python experiments/E_pos_eval.py --shuffle_context_window --input_mode "spatial" --encoding_method "latency" --sim_steps $sim_steps --batch_size $batch_size --model_path "output_results\E_pos\main\upos_2026-05-02_17-32-33_e-50_s-20_spatial.pt"
# python experiments/E_pos_eval.py --input_mode "temporal" --encoding_method "latency" --sim_steps $sim_steps --batch_size $batch_size --model_path "output_results\E_pos\main\upos_2026-05-03_07-46-23_e-50_s-20_temporal.pt"
# python experiments/E_pos_eval.py --shuffle_context_window --input_mode "temporal" --encoding_method "latency" --sim_steps $sim_steps --batch_size $batch_size --model_path "output_results\E_pos\main\upos_2026-05-03_07-46-23_e-50_s-20_temporal.pt"


# UPOS seq2seq mode
python experiments/E_pos_seq.py --save --eval --input_mode "spatial" --encoding_method "latency" --output_file_prefix "seq_tmp" --epochs 1 --beta $beta --alpha $alpha --sim_steps $sim_steps --limit 1000 --learning_rate $lr --batch_size 64
# python experiments/E_pos_seq.py --save --eval --input_mode "spatial" --encoding_method "latency" --output_file_prefix "seq" --epochs $epochs --beta $beta --alpha $alpha --sim_steps $sim_steps --limit $limit --learning_rate $lr --batch_size $batch_size

# UPOS ANN MLP
# python experiments/E_pos_ann-mlp.py --save --eval --limit 1000 --learning_rate $lr --batch_size $batch_size --epochs 1
# python experiments/E_pos_seq_ann-mlp.py --save --eval --limit 1000 --learning_rate $lr --batch_size $batch_size --epochs 1
