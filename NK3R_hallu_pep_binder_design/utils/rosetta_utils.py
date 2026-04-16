import json
import sys
import traceback
from pathlib import Path
from typing import List

PyRosetta_AVAILABLE = False
try:
    import pyrosetta
    pyrosetta.init(extra_options="-mute all -constant_seed")
    from pyrosetta import pose_from_pdb
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover, InterfaceRegion
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    PyRosetta_AVAILABLE = True
except ImportError:
    print("\n[WARNING] PyRosetta module import failed. Rosetta metrics cannot be calculated.")
except Exception as e:
    print(f"\n[WARNING] PyRosetta initialization failed: {e}. Rosetta metrics cannot be calculated.")

def _load_hotspot_residue_ids(hotspot_file_path: Path) -> set:
    """Loads a set of PDB residue numbers for known hotspots from a JSON file."""
    try:
        with open(hotspot_file_path, 'r') as f:
            data = json.load(f)
            # Data is expected format: [[PDB_NUM, "AMINO_ACID"], ...]
            return set(item[0] for item in data)
    except Exception as e:
        print(f"  - WARNING: Failed to load hotspot file {hotspot_file_path}: {e}")
        return set()

def quick_relax_pose(pose, scorefxn):
    """Performs a single round of constrained FastRelax."""
    if not PyRosetta_AVAILABLE: return pose

    try:
        tf = TaskFactory()
        # Restrict to repacking only
        tf.push_back(RestrictToRepacking())

        # FastRelax with 5 round
        fast_relax = FastRelax(scorefxn, 5)
        fast_relax.set_task_factory(tf)

        # Enforce constraints
        fast_relax.constrain_relax_to_start_coords(True)
        fast_relax.coord_constrain_sidechains(True)
        fast_relax.ramp_down_constraints(False) # Do not ramp down constraints

        fast_relax.apply(pose)
        return pose
    except Exception as e:
        print(f"    WARNING: Quick relax failed: {e}")
        return pose

def run_PyRosetta_interface_analysis(colabfold_output_dir: Path, peptide_chain_id: str, receptor_chain_ids: List, hotspot_file_path: Path, do_relax: bool = True):
    """
    Calculates interface metrics, automatically determining receptor chains and
    measuring hotspot coverage.

    Args:
        colabfold_output_dir (Path): Directory containing the predicted PDB file.
        peptide_chain_id (str): Chain ID of the peptide (e.g., 'A').
        receptor_chain_ids (List): Chain IDs of receptor (e.g., ['B', 'C']).
        hotspot_file_path (Path): Path to the JSON file containing known hotspot residue IDs.
        do_relax (bool): Whether to perform a quick sidechain relax before analysis.

    Returns:
        dict: A dictionary containing various Rosetta interface metrics, or an error dictionary.
    """
    # Placeholder checks for PyRosetta availability (replace with your actual variable)
    PyRosetta_AVAILABLE = ('pyrosetta' in sys.modules)

    if not PyRosetta_AVAILABLE:
        return {'error': 'PyRosetta_not_available'}

    # Locate PDB file
    pdb_files = list(colabfold_output_dir.glob('*_relaxed_rank_*.pdb'))
    if not pdb_files:
        print(f"  - ERROR: No relaxed PDB file found for Rosetta analysis.")
        return {'error': 'pdb_not_found'}
    complex_pdb = pdb_files[0]

    # Load Hotspots
    known_hotspot_ids = _load_hotspot_residue_ids(hotspot_file_path)
    total_known_hotspots = len(known_hotspot_ids)
    if total_known_hotspots == 0:
        print("  - WARNING: Loaded 0 known hotspot residues. Hotspot coverage will be 0%.")

    try:
        # Prepare pose and determine chains
        pose = pose_from_pdb(str(complex_pdb))
        receptor_str = "".join(receptor_chain_ids)
        interface_def = f"{receptor_str}_{peptide_chain_id}"
        if not interface_def:
            return {'error': 'invalid_interface_def'}

        # Initialize Score Function and Relax
        scorefxn = ScoreFunctionFactory.create_score_function("ref2015")
        energy_after_relax = scorefxn(pose) # Initial score

        if do_relax:
            print("  - Performing quick sidechain Relax...")
            pose = quick_relax_pose(pose, scorefxn)
            energy_after_relax = scorefxn(pose) # Score after relax

        # Run InterfaceAnalyzerMover
        interface_analyzer = InterfaceAnalyzerMover(interface_def)
        interface_analyzer.set_scorefunction(scorefxn)
        interface_analyzer.set_compute_interface_sc(True)
        interface_analyzer.set_compute_packstat(True)
        interface_analyzer.set_pack_separated(True)
        interface_analyzer.set_pack_input(True)
        print(f"  - Running InterfaceAnalyzerMover (Interface: {interface_def}, Receptor chains: {receptor_chain_ids})...")
        interface_analyzer.apply(pose)

        # Calculate hotspot coverage metric
        receptor_interface_pdb_ids = set()

        interface_residue_indices = interface_analyzer.get_interface_set()
        for i in interface_residue_indices:
            res_chain = pose.pdb_info().chain(i)
            if res_chain in receptor_chain_ids:
                res_num = pose.pdb_info().number(i)
                receptor_interface_pdb_ids.add(res_num)

        # Calculate overlap and percentage
        hotspot_overlap_ids = receptor_interface_pdb_ids.intersection(known_hotspot_ids)
        hotspot_overlap_count = len(hotspot_overlap_ids)

        if total_known_hotspots > 0:
            hotspot_coverage_percent = (hotspot_overlap_count / total_known_hotspots) * 100
        else:
            hotspot_coverage_percent = 0.0

        # Extract standard Metrics (existing logic)
        dG_separated = interface_analyzer.get_separated_interface_energy()
        dSASA_int_total = interface_analyzer.get_interface_delta_sasa()
        delta_unsat_hbonds = interface_analyzer.get_interface_delta_hbond_unsat()
        all_data = interface_analyzer.get_all_data()

        dSASA_polar = all_data.dhSASA[InterfaceRegion.total]
        dSASA_hydrophobic = dSASA_int_total - dSASA_polar
        sc_value = all_data.sc_value
        interface_hbonds = all_data.interface_hbonds
        packstat_value = all_data.packstat
        energy_efficiency = all_data.dG_dSASA_ratio
        interface_residues = interface_analyzer.get_num_interface_residues()

        # Compile Results
        results = {
            'interface_definition': interface_def,
            'dG_separated': dG_separated,
            'dSASA_int': dSASA_int_total,
            'dSASA_polar': dSASA_polar,
            'dSASA_hydrophobic': dSASA_hydrophobic,
            'delta_unsat_hbonds': delta_unsat_hbonds,
            'shape_complementarity_sc': sc_value,
            'interface_hbonds': interface_hbonds,
            'packstat': packstat_value,
            'dG_dSASA_ratio': energy_efficiency,
            'interface_residues': interface_residues,
            'complex_energy_after_relax': energy_after_relax,
            'receptor_hotspot_overlap_count': hotspot_overlap_count,
            'receptor_hotspot_coverage_percent': hotspot_coverage_percent,
        }

        if results['interface_residues'] < 5:
            print(f"  - WARNING: Very few interface residues found ({results['interface_residues']}).")
        # print(f"  - Hotspot Coverage: {hotspot_overlap_count}/{total_known_hotspots} ({hotspot_coverage_percent:.2f}%)")
        return results

    except Exception as e:
        print(f"  - CRITICAL ERROR: Rosetta analysis failed: {e}")
        traceback.print_exc()
        return {'error': 'pyrosetta_analysis_failed', 'detail': str(e)}


if __name__ == "__main__":
    colabfold_output_dir = Path("/home/qiuyk/code/Xpep/NK3R_Peptide_Binder/Xpep_iterative_cycle_results/cycle_1/colabfold_output")
    peptide_chain_id = 'A'
    receptor_chain_ids = ['B']
    hotspot_file_path = Path("/home/qiuyk/code/Xpep/NK3R_Peptide_Binder/analysis_results/hotspot_residues.json")
    run_PyRosetta_interface_analysis(colabfold_output_dir, peptide_chain_id, receptor_chain_ids, hotspot_file_path, do_relax = False)
