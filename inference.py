from typing import Any
import argparse
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import lightning
import numpy as np
import rootutils
import torch
from Bio.PDB import PDBParser
from Bio.PDB.NeighborSearch import NeighborSearch
from omegaconf import OmegaConf

from REAPS.data.constants import ATOM_ORDER, RESTYPE_3_TO_1, IDX_TO_AA
from REAPS.models.REAPS_model import REAPS_Model
from REAPS.models.featurizer import get_virtual_cb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
warnings.filterwarnings("ignore", "Ignoring unrecognized record 'END'", module="Bio.PDB.PDBParser")


def get_chain_order_from_pdb_file(pdb_path: Path) -> list[str]:
    chain_order = []
    seen = set()
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith(("ATOM  ", "HETATM")):
                    if len(line) >= 22:
                        chain_id = line[21].strip()
                        if chain_id == "":
                            chain_id = "_"
                        if chain_id not in seen:
                            seen.add(chain_id)
                            chain_order.append(chain_id)
    except Exception as e:
        print(f"Warning: failed to extract chain order from raw PDB file: {e}")

    return chain_order


def parse_pdb_to_features(pdb_path: str, peptide_chain_id: str, mode: str, cutoff_radius: float = 20.0):
    pdb_path = Path(pdb_path)
    pdb_id = pdb_path.stem
    parser = PDBParser(QUIET=True)

    # Preserve the actual file order directly from raw PDB text
    raw_chain_order = get_chain_order_from_pdb_file(pdb_path)

    try:
        structure = parser.get_structure(pdb_id, str(pdb_path))
    except Exception as e:
        print(f"Error when parsing PDB file {pdb_path}: {e}")
        return None

    model = structure[0]

    receptor_valid_residues = None
    if mode == "cyclic":
        peptide_chain = None
        for chain in model:
            if chain.id == peptide_chain_id:
                peptide_chain = chain
                break

        if peptide_chain is not None:
            binder_atoms = list(peptide_chain.get_atoms())
            ns = NeighborSearch(binder_atoms)
            receptor_valid_residues = set()

            for chain in model:
                if chain.id == peptide_chain_id:
                    continue

                for res in chain:
                    if res.id[0] != " ":
                        continue

                    target_coord = None
                    if "CB" in res:
                        target_coord = res["CB"].coord
                    elif "N" in res and "CA" in res and "C" in res:
                        n = torch.tensor(res["N"].coord, dtype=torch.float32)
                        ca = torch.tensor(res["CA"].coord, dtype=torch.float32)
                        c = torch.tensor(res["C"].coord, dtype=torch.float32)
                        v_cb = get_virtual_cb(n, ca, c)
                        target_coord = v_cb.numpy()
                    elif "CA" in res:
                        target_coord = res["CA"].coord

                    if target_coord is not None:
                        nearby_peptide_atoms = ns.search(target_coord, cutoff_radius)
                        if len(nearby_peptide_atoms) > 0:
                            receptor_valid_residues.add((chain.id, res.id))
        else:
            print(f"Warning: Macrocyclic Peptide chain '{peptide_chain_id}' not found. Skipping truncation.")

    chain_features_list = []
    full_structure_info = []
    ESSENTIAL_BACKBONE_ATOMS = ["N", "CA", "C", "O"]

    total_skipped_non_standard = 0

    for chain in model:
        seq = []
        full_seq = []
        res_indices = []
        coords_list = []
        mask_list = []
        skipped_residues = 0
        valid_residue_count = 0

        for res in chain:
            if res.id[0] != " ":
                continue

            res_name = res.get_resname()
            if res_name not in RESTYPE_3_TO_1:
                skipped_residues += 1
                total_skipped_non_standard += 1
                continue

            aa = RESTYPE_3_TO_1[res_name]
            full_seq.append(aa)

            backbone_complete = True
            atom_names = {atom.get_name() for atom in res}
            for atom_name in ESSENTIAL_BACKBONE_ATOMS:
                if atom_name not in atom_names:
                    backbone_complete = False
                    break

            if not backbone_complete:
                print(
                    f"Warning: Residue {res.id[1]} ({res_name}) in chain {chain.id} "
                    f"missing essential backbone atoms. Skipping for inference."
                )
                skipped_residues += 1
                total_skipped_non_standard += 1
                continue

            if receptor_valid_residues is not None and chain.id != peptide_chain_id:
                if (chain.id, res.id) not in receptor_valid_residues:
                    continue

            original_res_id = res.id[1]
            seq.append(aa)
            res_indices.append(original_res_id)
            valid_residue_count += 1

            xyz_37 = np.full((37, 3), np.nan, dtype=np.float32)
            xyz_37_mask = np.zeros(37, dtype=np.float32)
            for atom in res:
                atom_name = atom.get_name()
                if atom_name in ATOM_ORDER:
                    idx = ATOM_ORDER[atom_name]
                    xyz_37[idx] = atom.get_coord()
                    xyz_37_mask[idx] = 1.0

            if np.sum(xyz_37_mask) == 0:
                print(
                    f"Warning: No valid atoms found for residue {res.id[1]} ({res_name}) "
                    f"in chain {chain.id}. Skipping for inference."
                )
                seq.pop()
                res_indices.pop()
                valid_residue_count -= 1
                skipped_residues += 1
                total_skipped_non_standard += 1
                continue

            coords_list.append(xyz_37)
            mask_list.append(xyz_37_mask)

        full_structure_info.append({
            "chain_id": chain.id,
            "full_seq": "".join(full_seq),
            "original_chain_length": len(chain),
        })

        if skipped_residues > 0:
            print(
                f"Info: Skipped {skipped_residues} residues in chain {chain.id} "
                f"due to missing atoms or non-standard residues."
            )

        if len(seq) == 0:
            if chain.id == peptide_chain_id:
                print(f"Error: Target peptide chain {chain.id} has no valid residues for inference.")
                return None
            continue

        chain_features_list.append({
            "chain_id": chain.id,
            "seq": "".join(seq),
            "full_seq": "".join(full_seq),
            "xyz_37": np.stack(coords_list),
            "xyz_37_mask": np.stack(mask_list),
            "R_idx": np.array(res_indices, dtype=np.int32),
            "original_chain_length": len(chain),
            "valid_residue_count": valid_residue_count,
        })

    if not chain_features_list:
        print(f"Error: No valid protein chains found in {pdb_path}.")
        return None

    for chain_feat in chain_features_list:
        seq_len = len(chain_feat["seq"])
        coord_len = chain_feat["xyz_37"].shape[0]
        mask_len = chain_feat["xyz_37_mask"].shape[0]
        idx_len = len(chain_feat["R_idx"])
        if not (seq_len == coord_len == mask_len == idx_len):
            print(f"Error: Dimension mismatch in chain {chain_feat['chain_id']}")
            return None

    parsed_chain_ids = {c["chain_id"] for c in full_structure_info}

    chain_order = [cid for cid in raw_chain_order if cid in parsed_chain_ids]

    if not chain_order:
        chain_order = [c["chain_id"] for c in full_structure_info]

    data_dict: dict[str, Any] = {
        "chain_features": chain_features_list,
        "full_structure_info": full_structure_info,
        "chain_order": chain_order,
        "pdb_id": pdb_id,
        "parsing_info": {
            "total_chains_processed": len(full_structure_info),
            "total_chains_used_for_inference": len(chain_features_list),
            "skipped_non_standard": total_skipped_non_standard,
        },
    }
    return data_dict


def calculate_peptide_recovery(native_peptide_seq: str, designed_peptide_seq: str) -> float:
    if len(native_peptide_seq) != len(designed_peptide_seq):
        print("Warning: Native and designed peptide sequences have different lengths, Recovery set to 0.0!")
        return 0.0
    if not native_peptide_seq:
        return 0.0
    matches = sum(1 for native, designed in zip(native_peptide_seq, designed_peptide_seq) if native == designed)
    return matches / len(native_peptide_seq)


def assemble_sample_data(outputs):
    all_data: dict[str, Any] = defaultdict(lambda: {"designed_sequences": [], "log_likelihoods": []})

    for result in outputs:
        sampled_tokens = result.get("all_sampled_tokens", None)
        if sampled_tokens is None or len(sampled_tokens) == 0:
            continue

        y_true_batch = result["y_true"]
        sampled_tokens_batch = result["all_sampled_tokens"]
        sampled_log_likelihoods = result.get("all_sampled_log_likelihoods", [])
        batch_ids = result["batch_id_peptide"]
        unique_keys_batch_map = result["unique_keys_batch_map"]

        unique_ids, counts = np.unique(batch_ids, return_counts=True)
        y_true_peptides = np.split(y_true_batch, np.cumsum(counts)[:-1])
        K = len(sampled_tokens_batch)
        sampled_peptides_k_list = [np.split(s_i, np.cumsum(counts)[:-1]) for s_i in sampled_tokens_batch]

        for i, local_batch_id in enumerate(unique_ids):
            unique_key = unique_keys_batch_map[local_batch_id]
            pdb_id, chain_id = unique_key.rsplit("_", 1)

            gt_sequence_indices = y_true_peptides[i].tolist()
            gt_sequence_str = "".join([IDX_TO_AA.get(idx, "X") for idx in gt_sequence_indices])

            designed_sequences_str = []
            current_log_likelihoods = []

            for k in range(K):
                designed_indices = sampled_peptides_k_list[k][i].tolist()
                designed_seq = "".join([IDX_TO_AA.get(idx, "X") for idx in designed_indices])
                designed_sequences_str.append(designed_seq)

                if sampled_log_likelihoods is not None and len(sampled_log_likelihoods) > 0:
                    current_log_likelihoods.append(sampled_log_likelihoods[k])

            all_data[unique_key]["unique_key"] = unique_key
            all_data[unique_key]["pdb_id"] = pdb_id
            all_data[unique_key]["chain_id"] = chain_id
            all_data[unique_key]["native_sequence"] = gt_sequence_str
            all_data[unique_key]["designed_sequences"].extend(designed_sequences_str)
            all_data[unique_key]["log_likelihoods"].extend(current_log_likelihoods)

    return list(all_data.values())


def write_fasta(
        assembled_data: list,
        data_dict: dict,
        target_chain_id: str,
        output_path: str,
        pdb_file_name: str,
        temperature: float,
        checkpoint_path: str,
):
    full_pdb_path = str(Path(pdb_file_name).resolve())
    pdb_stem = Path(pdb_file_name).stem
    target_key = f"{pdb_stem}_{target_chain_id}"

    entry = None
    for d in assembled_data:
        if d.get("unique_key") == target_key:
            entry = d
            break

    if entry is None:
        print(f"[write_fasta] Cannot find entry for target_key={target_key}")
        return

    all_chains_fasta = data_dict.get("full_structure_info", [])
    chain_order = data_dict.get("chain_order", [])

    if not all_chains_fasta:
        print("[write_fasta] data_dict['full_structure_info'] is empty.")
        return

    chain_info_map = {c["chain_id"]: c for c in all_chains_fasta}

    # Fallback
    if not chain_order:
        chain_order = [c["chain_id"] for c in all_chains_fasta]

    # Keep only chains that actually exist
    chain_order = [cid for cid in chain_order if cid in chain_info_map]

    native_peptide_seq = None
    native_seq_list = []

    for cid in chain_order:
        seq = chain_info_map[cid]["full_seq"]
        native_seq_list.append(seq)
        if cid == target_chain_id:
            native_peptide_seq = seq

    if native_peptide_seq is None:
        print(f"[write_fasta] Cannot find native peptide sequence for chain {target_chain_id}")
        return

    full_native_sequence = ":".join(native_seq_list)
    full_designed_sequences_and_confidences = []
    designed_peptide_seqs = entry["designed_sequences"]
    log_likelihoods = entry.get("log_likelihoods", [None] * len(designed_peptide_seqs))

    L = len(native_peptide_seq)
    for designed_peptide_seq, log_likelihood in zip(designed_peptide_seqs, log_likelihoods):
        recovery_rate = calculate_peptide_recovery(native_peptide_seq, designed_peptide_seq)
        current_full_seq_list = []

        for cid in chain_order:
            if cid == target_chain_id:
                current_full_seq_list.append(designed_peptide_seq)
            else:
                current_full_seq_list.append(chain_info_map[cid]["full_seq"])

        full_designed_seq = ":".join(current_full_seq_list)

        avg_confidence = None
        if log_likelihood is not None and L > 0:
            avg_log_likelihood = log_likelihood / L
            avg_confidence = math.exp(avg_log_likelihood)

        full_designed_sequences_and_confidences.append({
            "full_seq": full_designed_seq,
            "recovery": recovery_rate,
            "avg_confidence": avg_confidence,
        })

    output_path = Path(output_path) / "seqs" / f"{pdb_stem}.fa"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(
            f">native_{pdb_stem} | {full_pdb_path} | chain_order={':'.join(chain_order)}\n"
            f"{full_native_sequence}\n"
        )
        for i, item in enumerate(full_designed_sequences_and_confidences):
            conf_str = f" | confidence={item['avg_confidence']:.4f}" if item["avg_confidence"] is not None else ""
            header = (
                f">sample_{i+1}_{pdb_stem} | recovery={item['recovery']:.4f}{conf_str} "
                f"| {full_pdb_path} | temperature={temperature} | checkpoint={checkpoint_path} "
                f"| chain_order={':'.join(chain_order)}"
            )
            f.write(f"{header}\n{item['full_seq']}\n")

    print(f"\nSuccess! Sampled sequences saved to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="--------------- //REAPS INFERENCE SCRIPT ARGS// ---------------")
    parser.add_argument("--pdb_file", type=str, default="example/1T79.pdb",
                        help="Path to input PDB file")
    parser.add_argument("--peptide_chain_id", type=str, default="B",
                        help="Chain ID to be treated as the peptide")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/REAPS_n0.02_pepFT.ckpt",
                        help="Path to model weights")
    parser.add_argument("--model_config_path", type=str, default="configs/model/REAPS.yaml")
    parser.add_argument("--fasta_output_path", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=3407,
                        help="Random seed for reproducibility.")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mode", type=str, default="linear", choices=["linear", "cyclic"],
                        help="Peptide mode: 'linear' or 'cyclic'")
    args = parser.parse_args()

    lightning.seed_everything(args.seed)

    if not Path(args.model_config_path).exists():
        print(f"Error! Model config file not found: {args.model_config_path}")
        sys.exit(1)

    model_params = OmegaConf.load(args.model_config_path)

    ckpt_name = Path(args.checkpoint_path).name
    is_mode_cyclic = args.mode == "cyclic"
    is_ckpt_cyclic = "cyclic" in ckpt_name.lower()

    if is_mode_cyclic and not is_ckpt_cyclic:
        print(
            f"WARNING: '--mode cyclic' is specified, but the checkpoint '{ckpt_name}' "
            f"does not contain the 'cyclic' identifier. Mode and weights may mismatch!"
        )
    elif not is_mode_cyclic and is_ckpt_cyclic:
        print(
            f"WARNING: '--mode linear' is specified, but the checkpoint '{ckpt_name}' "
            f"contains the 'cyclic' identifier. Mode and weights may mismatch!"
        )

    model_params.mode = args.mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {Path(args.checkpoint_path).name}")

    try:
        model: REAPS_Model = REAPS_Model.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            **model_params,
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    model = model.to(device).eval()
    model.featurizer.to(device)

    data_dict = parse_pdb_to_features(
        pdb_path=args.pdb_file,
        peptide_chain_id=args.peptide_chain_id,
        mode=model_params.mode,
        cutoff_radius=20.0,
    )
    if data_dict is None:
        sys.exit(1)

    raw_batch = [data_dict]
    chain_ids_for_inference = [args.peptide_chain_id]
    target_chain_id_for_output = args.peptide_chain_id

    print(
        f"Starting sampling for chain '{target_chain_id_for_output}', "
        f"temperature {args.temperature}, {args.num_samples} sequences..."
    )

    with torch.no_grad():
        try:
            results = model(raw_batch, inference_peptide_chain_ids=chain_ids_for_inference)
            if results is None or "batched_graph" not in results:
                print("Error: Model forward pass returned no valid results.")
                sys.exit(1)

            sampling_results = model.test_sample_peptide_sequences(
                raw_batch,
                sample_temperature=args.temperature,
                num_samples=args.num_samples,
                inference_peptide_chain_ids=chain_ids_for_inference,
            )

            peptide_mask = results["batched_graph"]["peptide_mask"]
            batch_id = results["batched_graph"]["batch_id"]

            sampling_results["batch_id_peptide"] = batch_id[peptide_mask].cpu().numpy()
            sampling_results["unique_keys_batch_map"] = [f"{data_dict['pdb_id']}_{target_chain_id_for_output}"]

            outputs_list = [sampling_results]
            assembled_data = assemble_sample_data(outputs_list)

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    if assembled_data:
        write_fasta(
            assembled_data,
            data_dict,
            target_chain_id_for_output,
            args.fasta_output_path,
            args.pdb_file,
            args.temperature,
            args.checkpoint_path,
        )
    else:
        print("Error: No assembled data after sampling.")


if __name__ == "__main__":
    main()