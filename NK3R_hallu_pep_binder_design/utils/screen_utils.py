import json
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.PDBExceptions import PDBConstructionWarning


def analyze_and_filter_predictions(base_dir, hotspot_file, receptor_chain_ids,
                                   peptide_chain_id, clash_threshold, max_clashes,
                                   hotspot_contact_dist, hotspot_coverage_thresh, yaml_path):
    print(f"\n====== Analyzing and Filtering Prediction ======")
    try:
        with open(hotspot_file, 'r') as f:
            hotspot_info = json.load(f)
        hotspot_residue_ids = set(item[0] for item in hotspot_info)
        total_known_hotspots = len(hotspot_residue_ids)
        print(f"Successfully loaded {total_known_hotspots} hotspot residue IDs.")
    except FileNotFoundError:
        print(f"Error: Hotspot file not found at {hotspot_file}")
        return None

    required_hotspot_contacts = np.ceil(total_known_hotspots * hotspot_coverage_thresh / 100.0)

    print(f"------- Filtering Thresholds -------")
    print(f"Required Hotspot Contacts: >= {required_hotspot_contacts} (>= {hotspot_coverage_thresh}% of {total_known_hotspots})")
    print(f"Allowed Clashes: <= {max_clashes}")
    print("-------------------------------------")

    parser = PDBParser(QUIET=True)
    RUN_PREDICTIONS_SUBDIR = f"boltz_results_{yaml_path.stem}/predictions/{yaml_path.stem}"
    single_pdb_path = None
    unique_id = None

    try:
        predictions_base = base_dir / RUN_PREDICTIONS_SUBDIR
        if not predictions_base.exists():
            print(f"Error: Predictions directory not found at {predictions_base}")
            return None

        print(f"Searching for a single PDB file in: {predictions_base}")

        pdb_files_found = list(predictions_base.glob('*.pdb'))

        if len(pdb_files_found) == 0:
            print(f"Warning: Found 0 PDB files in {predictions_base}")
            return None

        if len(pdb_files_found) > 1:
            print(f"Warning: Found {len(pdb_files_found)} PDBs, but expected only 1. Using the first one found.")

        single_pdb_path = pdb_files_found[0]
        unique_id = f"{predictions_base.name}/{single_pdb_path.name}"

    except FileNotFoundError:
        print(f"Error: Predictions directory not found at {predictions_base}")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)

        print(f"--- Processing: {unique_id} ---")
        clash_count = -1
        hotspot_contacts = -1
        passes_filter = False

        try:
            structure = parser.get_structure('candidate', single_pdb_path)
            model = structure[0]
            all_chains_present = True
            if peptide_chain_id not in model: all_chains_present = False
            for rec_id in receptor_chain_ids:
                if rec_id not in model: all_chains_present = False
            if not all_chains_present:
                print(f"  - Warning: Missing chains. Skipping.")
                return None

            peptide_chain = model[peptide_chain_id]

            backbone_atoms = {'N', 'CA', 'C', 'O'}
            atoms_to_remove = []
            for res in peptide_chain.get_residues():
                for atom in res.get_atoms():
                    if atom.get_id() not in backbone_atoms:
                        atoms_to_remove.append((res, atom.get_id()))
            for res, atom_id in atoms_to_remove:
                res.detach_child(atom_id)

            peptide_backbone_atoms = list(peptide_chain.get_atoms())
            receptor_all_atoms = []
            for rec_id in receptor_chain_ids:
                receptor_all_atoms.extend(list(model[rec_id].get_atoms()))

            if not receptor_all_atoms or not peptide_backbone_atoms:
                print("  - Warning: Empty chains after processing. Skipping.")
                return None

            clash_search_atoms = receptor_all_atoms + peptide_backbone_atoms
            ns_clash = NeighborSearch(clash_search_atoms)
            all_atom_pairs = ns_clash.search_all(radius=clash_threshold, level='A')
            clash_count = 0
            for a1, a2 in all_atom_pairs:
                c1 = a1.get_parent().get_parent().id
                c2 = a2.get_parent().get_parent().id
                if (c1 in receptor_chain_ids and c2 == peptide_chain_id) or \
                        (c1 == peptide_chain_id and c2 in receptor_chain_ids):
                    clash_count += 1

            hotspot_atoms = []
            for rec_id in receptor_chain_ids:
                for res in model[rec_id].get_residues():
                    if res.id[1] in hotspot_residue_ids:
                        hotspot_atoms.extend(res.get_atoms())

            peptide_ca_atoms = [a for a in peptide_backbone_atoms if a.get_id() == 'CA']

            hotspot_contact_residues = set()

            if hotspot_atoms and peptide_ca_atoms:
                ns_hotspot = NeighborSearch(hotspot_atoms + peptide_ca_atoms)
                all_hotspot_pairs = ns_hotspot.search_all(radius=hotspot_contact_dist, level='A')

                for a1, a2 in all_hotspot_pairs:
                    res_rec = None
                    c1 = a1.get_parent().get_parent().id
                    c2 = a2.get_parent().get_parent().id

                    if c1 in receptor_chain_ids and c2 == peptide_chain_id:
                        res_rec = a1.get_parent()
                    elif c1 == peptide_chain_id and c2 in receptor_chain_ids:
                        res_rec = a2.get_parent()
                    if res_rec:
                        res_key = (res_rec.get_parent().id, res_rec.id)
                        hotspot_contact_residues.add(res_key)
                hotspot_contacts = len(hotspot_contact_residues)
            else:
                hotspot_contacts = 0

            passes_clash = clash_count <= max_clashes
            passes_hotspot = hotspot_contacts >= required_hotspot_contacts
            passes_filter = passes_clash and passes_hotspot

            print(f"  - Clash check: {clash_count} pairs (Pass: {passes_clash})")
            print(f"  - Hotspot report: {hotspot_contacts} unique contacts (Pass: {passes_hotspot})")
            print(f"  - OVERALL: {'PASS' if passes_filter else 'FAIL'}")

        except Exception as e:
            print(f"  - ERROR processing {unique_id}: {e}")
            passes_filter = False

    if passes_filter:
        return single_pdb_path
    else:
        print(f"===== FAILURE =====")
        print(f"Candidate {unique_id} did not pass the filters.")
        return None

def extract_colabfold_metrics(colabfold_output_dir: Path, X_length: int) -> dict | None:
    scores_files = list(colabfold_output_dir.glob('*_scores_rank_*.json'))
    if not scores_files:
        print(f"  - Error: Scores JSON not found in {colabfold_output_dir.name}. Skipping.")
        return None
    scores_file_path = scores_files[0]

    pae_files = list(colabfold_output_dir.glob('*_predicted_aligned_error_v1.json'))
    pae_file_path = pae_files[0] if pae_files else None

    pdb_files = list(colabfold_output_dir.glob('*_relaxed_rank_*.pdb'))
    if not pdb_files:
        print(f"  - Warning: Relaxed PDB file not found. Skipping metric extraction.")
        return None
    predicted_pdb_path = pdb_files[0]

    metrics = {"pTM": 0.0, "ipTM": 0.0, "pLDDT": 0.0, "mean_interface_pAE": 0.0, "predicted_pdb_path": predicted_pdb_path}
    pae_matrix = None
    pae_matrix_scores = None

    try:
        with open(scores_file_path, 'r') as f:
            data = json.load(f)

            metrics['pTM'] = data.get('ptm', 0.0)
            metrics['ipTM'] = data.get('iptm', 0.0)

            plddt_values = data.get('plddt')
            pae_matrix_scores = data.get('pae')

            if plddt_values:
                metrics['pLDDT'] = np.mean(plddt_values)

    except Exception as e:
        print(f"  - Error reading Scores JSON {scores_file_path.name}: {e}")
        return None

    if pae_file_path:
        try:
            with open(pae_file_path, 'r') as f:
                pae_json_content = json.load(f)
                pae_matrix = pae_json_content.get('predicted_aligned_error')
        except Exception:
            if pae_matrix_scores is not None:
                pae_matrix = pae_matrix_scores
    elif pae_matrix_scores is not None:
        pae_matrix = pae_matrix_scores

    if pae_matrix is not None:
        pae_array = np.array(pae_matrix)
        N = pae_array.shape[0]
        N_peptide = X_length

        if N < N_peptide or N == 0:
            print("  - Error: pAE matrix size is inconsistent with expected peptide length. Skipping ipAE.")
        else:

            peptide_indices = slice(0, N_peptide)
            receptor_indices = slice(N_peptide, N)

            # Peptide x Receptor
            pae_pr = pae_array[peptide_indices, receptor_indices]

            # Receptor x Peptide
            pae_rp = pae_array[receptor_indices, peptide_indices]

            interface_pae_values = np.concatenate([pae_pr.flatten(), pae_rp.flatten()])

            if interface_pae_values.size > 0:
                metrics['mean_interface_pAE'] = np.mean(interface_pae_values)
            else:
                metrics['mean_interface_pAE'] = 0.0
                print("  - Warning: Interface pAE calculation resulted in 0 data points (Peptide == Receptor?).")

    if metrics['pTM'] == 0.0 and metrics['ipTM'] == 0.0 and metrics['pLDDT'] < 50.0:
        print("  - Warning: Core scores (pTM/ipTM) are 0.0 and pLDDT is low. Returning None.")
        return None

    return metrics
