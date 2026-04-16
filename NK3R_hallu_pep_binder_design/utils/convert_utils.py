import datetime
import json
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO


def format_time(seconds):
    """Formats seconds into H:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h-{m}m-{s}s"

def convert_unk_to_ala(input_pdb_path, output_pdb_path, peptide_chain_id) -> str | None:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("boltz_pdb", str(input_pdb_path))
        chain_found = False
        for chain in structure[0]:
            if chain.id == peptide_chain_id:
                chain_found = True
                for res in chain:
                    if res.get_resname() == 'UNK':
                        res.resname = 'ALA'
        if not chain_found:
            print(f"  - Warning: Peptide chain {peptide_chain_id} not found in {input_pdb_path}.")
            return None

        io = PDBIO()
        io.set_structure(structure)
        io.save(str(output_pdb_path))

        return output_pdb_path
    except Exception as e:
        print(f"  - Error during PDB conversion for {input_pdb_path}: {e}")
        return None

def save_metrics_to_json(colabfold_metrics: dict, rosetta_metrics: dict, output_dir: Path, i: int):
    """
    Saves all extracted metrics into a single JSON file, named after the predicted PDB.

    The output file will be named based on the PDB stem:
    e.g., design_00001_relaxed_rank_..._unified_metrics.json

    Args:
        colabfold_metrics (dict): ColabFold metrics, must contain 'predicted_pdb_path' (Path object).
        rosetta_metrics (dict): PyRosetta metrics.
        output_dir (Path): Base directory where the metrics JSON should be saved.
        i (int): cycle count.
    """

    # Validate and prepare path
    pdb_path_obj = colabfold_metrics.get('predicted_pdb_path')
    if not isinstance(pdb_path_obj, Path):
        print("CRITICAL ERROR: 'predicted_pdb_path' is missing or not a Path object. Cannot save.")
        return

    # Prepare unified data dictionary (convert Path to str for JSON serialization)
    temp_colabfold = colabfold_metrics.copy()
    temp_colabfold['predicted_pdb_path'] = str(temp_colabfold['predicted_pdb_path'])

    unified_metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "colabfold": temp_colabfold,
        "rosetta": rosetta_metrics,
        "notes": f"Unified metrics from iterative design loop, now in cycle {i}."
    }

    # Determine output filename
    # Use the PDB's stem (filename without extension)
    pdb_stem = pdb_path_obj.stem
    json_filename = f"{pdb_stem}_unified_metrics_cycle{i}.json"

    # Create output directory and save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = output_dir / json_filename

    try:
        with open(output_json_path, 'w') as f:
            json.dump(unified_metrics, f, indent=4)
        print(f"✓ Metrics successfully saved to: {output_json_path}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to save JSON file: {e}")
