from argparse import Namespace

# https://matplotlib.org/stable/gallery/color/named_colors.html
colors = Namespace(**{
    'train_accuracy': "royalblue",
    'test_accuracy': "blueviolet",
    'loss': "firebrick",
    'spike': "deeppink",
    'trace_line': "deepskyblue",
    'spike_dot': "indianred",
    'energy': "darkorange",
    'time': "limegreen",
    "fallback": "turquoise",
    "first_mean_time": "fuchsia"
})

def ModelNameConvention(s:str):
    s = s.replace('input_mode', 'poi-sc').replace('_', '-')
    s = s.replace('spatial', 'spa').replace('temporal', 'tem')
    s = s.replace('poisson', 'poi').replace('latency', 'lat')
    s = s.replace('-ce', '').replace('ce', '')
    return s