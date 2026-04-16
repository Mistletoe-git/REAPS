"""
Structural restoration for RFpeptides-generated complexes.

Because RFpeptides-generated complexes retain only receptor backbone atoms,
this script restores the full-atom receptor context required by REAPS.
Specifically, the generated receptor backbone is aligned to the native
reference receptor using Kabsch superposition, after which the original
all-atom receptor is combined with the newly generated peptide binder
backbone to reconstruct the final complex.
"""


import os
import glob
import copy
from Bio.PDB import PDBParser, Superimposer, PDBIO, Structure, Model


# Reference structure (The real receptor PDB containing full side-chains)
REF_PDB_PATH = "/path/to/your/reference/receptor.pdb"

# Support for multichain receptors: List all receptor chain IDs here
REF_RECEPTOR_CHAIN_IDS = ["A", "B"]

# Explicitly specify the cyclic peptide chain ID
GEN_PEPTIDE_CHAIN_ID = "A"   # The chain ID of the peptide in the RFpeptides generated PDB
OUT_PEPTIDE_CHAIN_ID = "A"   # The desired chain ID for the peptide in the final fixed PDB

# Input and output directories
INPUT_DIR = "/path/to/RFpeptides/outputs_dir"  # Directory containing RFpeptides outputs
OUTPUT_DIR = "/path/to/RFpeptides/fixed/outputs_dir"  # Directory for the fixed structures


def main():
    print("🔧 Starting global batch processing: [Multi-chain Alignment + Side-chain Restoration]...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = PDBParser(QUIET=True)
    io = PDBIO()

    # 1. Load and parse the reference structure
    print(f"[*] Loading reference structure: {REF_PDB_PATH} (Target Chains: {REF_RECEPTOR_CHAIN_IDS})")
    try:
        ref_struct = parser.get_structure("ref", REF_PDB_PATH)[0]
    except Exception as e:
        print(f"❌ Error: Failed to parse reference structure {REF_PDB_PATH}. Reason: {e}")
        return

    # Extract backbone atoms of the reference receptor across ALL specified chains
    ref_atoms = []
    for chain_id in REF_RECEPTOR_CHAIN_IDS:
        if chain_id not in ref_struct:
            print(f"❌ Error: Receptor chain '{chain_id}' not found in the reference structure!")
            return

        for res in ref_struct[chain_id]:
            if res.id[0] == ' ':
                if 'N' in res and 'CA' in res and 'C' in res:
                    ref_atoms.extend([res['N'], res['CA'], res['C']])

    print(f"    - Valid backbone atoms in reference receptor(s): {len(ref_atoms)}")

    # 2. Iteratively search for all PDB files in the input directory
    generated_pdbs = glob.glob(os.path.join(INPUT_DIR, "**", "*.pdb"), recursive=True)
    generated_pdbs = [f for f in generated_pdbs if os.path.isfile(f)]

    if not generated_pdbs:
        print(f"⚠️ Warning: No .pdb files found in {INPUT_DIR} or its subdirectories!")
        return

    print(f"\n[*] Scan complete. Found {len(generated_pdbs)} structures to process. Starting pipeline...\n")

    success_count = 0

    for idx, gen_pdb in enumerate(generated_pdbs):
        basename = os.path.basename(gen_pdb)
        final_pdb_path = os.path.join(OUTPUT_DIR, basename)

        try:
            mob_struct = parser.get_structure("mob", gen_pdb)[0]
        except Exception as e:
            print(f"  [!] Skipping {basename}: Failed to parse structure ({e})")
            continue

        # 3. Explicitly identify the peptide and collect all remaining chains as the generated receptor
        binder_chain = None
        mob_receptor_chains = []

        for chain in mob_struct:
            if chain.id == GEN_PEPTIDE_CHAIN_ID:
                binder_chain = chain
            else:
                mob_receptor_chains.append(chain)

        if not binder_chain:
            print(f"  [!] Skipping {basename}: Peptide chain '{GEN_PEPTIDE_CHAIN_ID}' not found in the generated file.")
            continue

        if not mob_receptor_chains:
            print(f"  [!] Skipping {basename}: No receptor chains found alongside the peptide.")
            continue

        # 4. Extract backbone atoms from all generated receptor chains
        mob_atoms = []
        for chain in mob_receptor_chains:
            for res in chain:
                if res.id[0] == ' ':
                    if 'N' in res and 'CA' in res and 'C' in res:
                        mob_atoms.extend([res['N'], res['CA'], res['C']])

        # Force truncation of both arrays to the same multiple of 3
        min_len = min(len(ref_atoms), len(mob_atoms))
        if min_len == 0:
            print(f"  [!] Skipping {basename}: Failed to extract valid backbone atoms for alignment.")
            continue

        min_len = min_len - (min_len % 3)
        final_ref_atoms = ref_atoms[:min_len]
        final_mob_atoms = mob_atoms[:min_len]

        # 5. Execute Kabsch structural superposition
        sup = Superimposer()
        sup.set_atoms(final_ref_atoms, final_mob_atoms)

        if idx < 5:
            print(f"      -> {basename} Alignment error (RMSD): {sup.rms:.4f} Å")
        elif idx == 5:
            print(f"      -> ...... (Subsequent file logs collapsed to keep output clean)")

        # 6. Apply the transformation matrix to the peptide
        sup.apply(binder_chain.get_atoms())

        # 7. [Architecture Reassembly] Create a pristine new complex model
        new_struct = Structure.Structure("final")
        new_model = Model.Model(0)

        # 7.1 Stitch in the reference receptor chains (preserving their original chain IDs and full side-chains)
        for chain_id in REF_RECEPTOR_CHAIN_IDS:
            target_receptor_chain = copy.deepcopy(ref_struct[chain_id])
            target_receptor_chain.detach_parent()
            new_model.add(target_receptor_chain)

        # 7.2 Stitch in the aligned peptide and assign it the configured output ID
        binder_chain_moved = copy.deepcopy(binder_chain)
        binder_chain_moved.detach_parent()
        binder_chain_moved.id = OUT_PEPTIDE_CHAIN_ID

        # Ensure no ID collision (e.g., if the user accidentally sets the peptide ID to match a receptor ID)
        if OUT_PEPTIDE_CHAIN_ID in new_model:
            print(f"  [!] Skipping {basename}: Output peptide chain ID '{OUT_PEPTIDE_CHAIN_ID}' conflicts with a receptor chain ID.")
            continue

        new_model.add(binder_chain_moved)

        # 8. Package and save
        new_struct.add(new_model)
        io.set_structure(new_struct)
        io.save(final_pdb_path)

        success_count += 1

    print(f"\n🎉 All structural alignments and side-chain restorations complete! Successfully fixed {success_count} files!")
    print(f"📁 Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()