import json
from pathlib import Path


def generate_boltz_input(receptor_seq_str: str, peptide_length: int, output_file_path: Path, is_cyclic: bool = False):
    """
    Generates a YAML file for Boltz at the specified output_file_path.
    - Peptide (Chain A): 'X' * peptide_length
    - Receptor (Chain B, C, ...): From receptor_seq_str
    - is_cyclic: If True, adds 'cyclic: true' to the peptide chain.
    """
    try:
        # Get the directory from the full path
        output_dir = output_file_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory confirmed: {output_dir}")
    except OSError as e:
        print(f"Failed to create directory {output_dir}: {e}")
        return None

    # Ensure the output file has a .yaml extension
    if output_file_path.suffix != '.yaml':
        output_file_path = output_file_path.with_suffix('.yaml')

    # Construct the YAML content manually to avoid requiring PyYAML dependency
    yaml_lines = [
        "version: 1",
        "sequences:",
        "  - protein:",
        "      id: A",
        f"      sequence: {'X' * peptide_length}"
    ]

    # Add cyclic flag if required
    if is_cyclic:
        yaml_lines.append("      cyclic: true")

    yaml_lines.append("      msa: empty")

    # Process Receptor Chains
    receptor_sequences = receptor_seq_str.split(':')
    receptor_chain_ids = []

    for i, seq in enumerate(receptor_sequences):
        chain_id = chr(ord('B') + i)
        receptor_chain_ids.append(chain_id)
        yaml_lines.append("  - protein:")
        yaml_lines.append(f"      id: {chain_id}")
        yaml_lines.append(f"      sequence: {seq.strip()}")
        yaml_lines.append("      msa: empty")

    # Combine into final string
    content = "\n".join(yaml_lines) + "\n"
    file_name = output_file_path.name

    try:
        with open(output_file_path, 'w') as f:
            f.write(content)
        print(f"  -> Generated: {file_name}")
        print(f"     Chains: A (Peptide, L={peptide_length}, cyclic={is_cyclic}), " +
              f"{', '.join(receptor_chain_ids)} (Receptor)")
    except IOError as e:
        print(f"  -> Failed to write file {file_name}: {e}")
        return None

    print("\nBoltz YAML input file generation complete.")

    return output_file_path

def extract_Boltz_metrics(metrics_dir: Path) -> dict | None:

    print(f"  - Searching for Boltz-2 metrics in: {metrics_dir}")

    if not metrics_dir.exists():
        print(f"  - !!!!! Error: Metrics directory not found.")
        return None

    try:
        json_files = list(metrics_dir.glob("confidence_*.json"))
        if not json_files:
            print(f"  - !!!!! Error: No 'confidence_*.json' file found in directory.")
            return None

        metrics_file_path = json_files[0]

        with open(metrics_file_path, 'r') as f:
            data = json.load(f)

        boltz_metrics = {
            "iptm": data.get("iptm"),
            "complex_plddt": data.get("complex_plddt"),
            "complex_iplddt": data.get("complex_iplddt")
        }

        if not all(v is not None for v in boltz_metrics.values()):
            missing = [k for k, v in boltz_metrics.items() if v is None]
            print(f"  - !!!!! Error: Metrics file is missing keys: {missing}")
            return None

        print(f"  - ✓ Boltz metrics extracted (ipTM: {boltz_metrics['iptm']:.4f}, pLDDT: {boltz_metrics['complex_plddt']:.4f}), ipLDDT: {boltz_metrics['complex_iplddt']:.4f}")
        return boltz_metrics

    except json.JSONDecodeError:
        print(f"  - !!!!! Error: Failed to decode JSON from {metrics_file_path}.")
        return None
    except Exception as e:
        print(f"  - !!!!! Error during metric extraction: {e}")
        return None
