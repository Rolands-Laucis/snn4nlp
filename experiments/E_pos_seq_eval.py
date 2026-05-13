import argparse
from pathlib import Path

from E_pos_seq_shared import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved SNN sequence POS model checkpoint")
    parser.add_argument("--model_path", type=str, required=True, help="Path to a saved .pt checkpoint")
    parser.add_argument("--input_file_prefix", type=str, default="pos_d100", help="Prefix for cast POS input files")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick evaluations")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override (defaults to checkpoint config)")
    parser.add_argument("--sim_steps", type=int, default=None, help="Simulation steps override (defaults to checkpoint config)")
    parser.add_argument("--input_mode", type=str, default=None, choices=["spatial", "temporal"], help="Input mode override")
    parser.add_argument("--encoding_method", type=str, default=None, choices=["poisson", "latency"], help="Encoding method override")
    parser.add_argument("--estimate_energy", action="store_true", help="Estimate average AC operations and energy per tested sample")
    parser.add_argument("--energy_ac_cost_pj", type=float, default=25.63, help="Energy cost of one AC operation in pJ (hardware-dependent)")
    parser.add_argument("--diagnose", action="store_true", help="Show spike trains and output-layer membrane trace for first sample")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save evaluation results as JSON")

    args = parser.parse_args()

    if not args.output_json:
        args.output_json = Path(args.model_path).parent / f"eval_{Path(args.model_path).stem}.json"

    evaluate_model(args)


if __name__ == "__main__":
    main()
