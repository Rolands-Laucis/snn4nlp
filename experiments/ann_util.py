def LSTM_FLOP_Estimate(batch_size, hidden_size, input_size, sequence_length=1):
    """
    Calculate FLOP count for LSTM computations per sample (per layer, but i only have 1 layer in my LSTM models).
    
    Formula: FLOPs = [8BH² + 8BIH + 9BH] × T
    where:
        - B: batch size
        - H: hidden size (number of units)
        - I: input dimension
        - T: sequence length (timesteps)
    
    Args:
        batch_size: Batch size (B)
        hidden_size: Number of LSTM units (H)
        input_size: Input dimension (I)
        sequence_length: Sequence length in timesteps (T)
    
    Returns:
        Total FLOPs per batch
    """
    B = batch_size
    H = hidden_size
    I = input_size
    T = sequence_length
    
    # Calculate FLOPs: [8BH² + 8BIH + 9BH] × T
    flops_per_timestep = (8 * B * H * H) + (8 * B * I * H) + (9 * B * H)
    total_flops = flops_per_timestep * T
    
    return total_flops

def LSTM_Energy_Estimate(flops, mode='cpu'):
    """
    Estimate energy consumption for LSTM computations.
    
    Converts FLOPs to energy using marginal energy per operation costs:
    - CPU: 744 J/GFLOP
    - GPU: 890 J/GFLOP
    
    Formula: Energy (J) = FLOPs × (Cost_per_GFLOP / 10^9 FLOP/GFLOP)
    Result converted to picojoules (1 J = 10^12 pJ)
    
    Args:
        flops: Total FLOPs for the LSTM computations
        mode: 'cpu' or 'gpu' (default: 'cpu')
    
    Returns:
        Total energy in picojoules
    """
    # Energy cost per GFLOP
    costs_per_gflop = {
        'cpu': 744,    # J/GFLOP
        'gpu': 890     # J/GFLOP
    }
    
    if mode not in costs_per_gflop:
        raise ValueError(f"Invalid mode: {mode}. Use 'cpu' or 'gpu'.")
    
    # joules per FLOP = joules per GFLOP / 1e9
    # pJ per FLOP = ((J/GFLOP) / 1e9) * 1e12 = (J/GFLOP) * 1e3
    pj_per_flop = costs_per_gflop[mode] * 1e3

    return pj_per_flop

def calculate_ann_total_energy(neurons_per_layer, emac_pj=57.14):
    """
    Calculate total energy for an ANN using MAC operations.
    
    Formula: E_ANN_total = Σ(N_{l-1} × N_l) × E_MAC
    where:
        - N_{l-1}: number of neurons in layer l-1
        - N_l: number of neurons in layer l
        - E_MAC: energy cost of a single MAC operation
    
    Args:
        neurons_per_layer: List of neuron counts for each layer [N_0, N_1, N_2, ..., N_L]
        emac_pj: Energy cost of a single MAC operation in picojoules
    
    Returns:
        Total energy in picojoules
    """
    total_energy = 0
    
    # Iterate through each layer (starting from layer 1)
    for l in range(1, len(neurons_per_layer)):
        n_prev = neurons_per_layer[l - 1]  # N_{l-1}
        n_curr = neurons_per_layer[l]      # N_l
        macs = n_prev * n_curr              # MAC operations for this layer
        energy_contribution = macs * emac_pj  # Energy for this layer
        total_energy += energy_contribution
        # print(f"Layer {l}: {n_prev} × {n_curr} = {macs} MACs → {energy_contribution:.2f} pJ")
    
    return total_energy

assert LSTM_FLOP_Estimate(1, 1, 1, 1) == 25 # Should be 25 FLOPs

if __name__ == "__main__":
    for dim in [25, 50, 100]:
        lstm_flops = LSTM_FLOP_Estimate(batch_size=1, hidden_size=256+128, input_size=dim, sequence_length=1)
        # emac_energy_pj = 57.14  # Energy per MAC operation in picojoules from SNNLP paper
        emac_energy_pj = LSTM_Energy_Estimate(lstm_flops, mode='cpu')
        neurons_per_layer = [dim*10, 256, 128, 2]
        total_energy = calculate_ann_total_energy(neurons_per_layer, emac_energy_pj) / 1e3 # convert from pJ to nJ
        print(f"Total ANN-{dim} cpu energy per sample: {total_energy:.0f} nJ") #, lstm_flops, emac_energy_pj