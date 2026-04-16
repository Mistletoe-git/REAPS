import json
import os
import warnings
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.PDBExceptions import PDBConstructionWarning


def define_hotspot(known_complex_pdbs, receptor_chain_ids, peptide_chain_ids, contact_distance, output_hotspot_file):
    parser = PDBParser(QUIET=True)
    hotspot_residue_info = set()
    print(f"Extracting hotspots from {len(known_complex_pdbs)} known complexes...")
    print(f"Contact logic: Peptide chain ({peptide_chain_ids}) 'CA' atoms <--> Receptor chain(s) ({receptor_chain_ids}) 'All' atoms, Distance < {contact_distance} Å")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        for pdb_file in known_complex_pdbs:
            try:
                print(f"  - Processing: {pdb_file}")
                structure = parser.get_structure('known', pdb_file)
                model = structure[0]

                all_chains_present = True
                if peptide_chain_ids not in model:
                    all_chains_present = False
                for rec_chain_id in receptor_chain_ids:
                    if rec_chain_id not in model:
                        all_chains_present = False

                if not all_chains_present:
                    print(f"    Warning: Peptide chain {peptide_chain_ids} or one or more receptor chains {receptor_chain_ids} not found.")
                    continue

                receptor_atoms = []
                for rec_chain_id in receptor_chain_ids:
                    receptor_atoms.extend(list(model[rec_chain_id].get_atoms()))

                peptide_chain = model[peptide_chain_ids]
                peptide_ca_atoms = []
                for atom in peptide_chain.get_atoms():
                    if atom.get_id() == 'CA':
                        peptide_ca_atoms.append(atom)
                print(f"    - Found {len(receptor_atoms)} receptor atoms (from {len(receptor_chain_ids)} chain(s)) and {len(peptide_ca_atoms)} peptide CA atoms.")

                if not receptor_atoms or not peptide_ca_atoms:
                    print(f"    Warning: Receptor atoms or peptide CA atoms list is empty.")
                    continue

                all_search_atoms = receptor_atoms + peptide_ca_atoms
                ns = NeighborSearch(all_search_atoms)
                all_atom_pairs = ns.search_all(radius=contact_distance, level='A')

                for a1, a2 in all_atom_pairs:
                    res1 = a1.get_parent()
                    res2 = a2.get_parent()
                    chain1_id = res1.get_parent().id
                    chain2_id = res2.get_parent().id
                    # Check for (receptor-peptide) or (peptide-receptor) contact
                    if chain1_id in receptor_chain_ids and chain2_id == peptide_chain_ids:
                        hotspot_residue_info.add( (res1.id[1], res1.resname) )
                    elif chain1_id == peptide_chain_ids and chain2_id in receptor_chain_ids:
                        hotspot_residue_info.add( (res2.id[1], res2.resname) )
                print(f"    Finished processing {pdb_file}.")
            except Exception as e:
                print(f"    Error processing {pdb_file}: {e}")

    hotspot_list_with_names = sorted(list(hotspot_residue_info))
    output_path = output_hotspot_file
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    with open(output_path, 'w') as f:
        json.dump(hotspot_list_with_names, f, indent=2)

    print(f"\nSuccessfully defined hotspots!")
    print(f"Found a total of {len(hotspot_list_with_names)} hotspot residues.")
    print(f"Hotspot residues (ID, Name): {hotspot_list_with_names}")
    print(f"Saved to {output_path}")