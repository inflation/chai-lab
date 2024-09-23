from pathlib import Path

import numpy as np
import torch

from chai_lab.chai1 import run_inference

# We use fasta-like format for inputs.
# - each entity encodes protein, ligand, RNA or DNA
# - each entity is labeled with unique name;
# - ligands are encoded with SMILES; modified residues encoded like AAA(SEP)AAA

# Example given below, just modify it

example_fasta = """
>protein|case1
STEYKLVVVGADGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEK
>ligand|chain-1
CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N(CC)[C@@H](CC3=CC=C(C)C=C3)C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H](CCC5)C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)[C@@H](CC(C)C)C(=O)N1)C(=O)N(C)C
>ligand|chain-2
CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N(CC)[C@@H](CC3=CC=C(C)C=C3)C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H](CCC5)C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)[C@@H](CC(C)C)C(=O)N1)C(=O)N8[C@@H](CCC8)C(=O)NCCCCCCC(=O)N[C@@H](C(C)(C)C)C(=O)N9[C@@H](C[C@@H](O)C9)C(=O)N[C@@H](C)C%10=CC=C(C=C%10)C%11=C(C)N=CS%11
>ligand|chain-3
CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N(CC)[C@@H](CC3=CC=C(C)C=C3)C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H](CCC5)C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)[C@@H](CC(C)C)C(=O)N1)C(=O)N8[C@@H](CCC8)C(=O)NCCCCC(=O)N[C@@H](C(C)(C)C)C(=O)N9[C@@H](C[C@@H](O)C9)C(=O)N[C@@H](C)C%10=CC=C(C=C%10)C%11=C(C)N=CS%11
>ligand|chain-4
CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N(CC)[C@@H](CC3=CC=C(C)C=C3)C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H](CCC5)C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)[C@@H](CC(C)C)C(=O)N1)C(=O)N8CCN(CC8)C(=O)CCCCCC(=O)N[C@@H](C(C)(C)C)C(=O)N9[C@@H](C[C@@H](O)C9)C(=O)N[C@@H](C)C%10=CC=C(C=C%10)C%11=C(C)N=CS%11
>ligand|chain-5
CC[C@H](C)[C@H]1C(=O)N(C)CC(=O)N(C)CC(=O)N(CC)[C@@H](CC2=CC=C(C)C=C2)C(=O)N(C)CC(=O)N[C@@H](CCC3=CC(F)=C(C(F)=C3)C(F)(F)F)C(=O)N4[C@@H](CCC4)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](C5CCCC5)C(=O)N(C)[C@H](CC(=O)N(C)[C@@H](CC(C)C)C(=O)N1)C(=O)N6CCN(C(=O)CCCCC(=O)N[C@@H](C(C)(C)C)C(=O)N7[C@@H](C[C@H](O)C7)C(=O)N[C@@H](C)C8=CC=C(C=C8)C9=C(C)N=CS9)CC6
>ligand|chain-6
CC[C@H](C)[C@H]1C(=O)N(C)CC(=O)N(C)CC(=O)N[C@H]2C(=O)N(C)CC(=O)N[C@@H](CCC3=CC(F)=C(C(F)=C3)C(F)(F)F)C(=O)N4[C@@H](CCC4)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](C5CCCC5)C(=O)N(C)[C@@H](CC(=O)NC(CN6C=C(N=N6)C7=CC=C(C=C7)C2)C(=O)N1)C(=O)N(C)C
>ligand|chain-7
CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N(CC)[C@@H](CC3=CC=C(C)C=C3)C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H](CCC5)C(=O)NC6(CC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)[C@@H](CC(C)C)C(=O)N1)C(=O)N(C)C
>ligand|chain-8
CC[C@H](C)[C@H]1C(=O)N(C)CC(=O)N(C)CC(=O)N(CC)[C@@H](CC2=CC=C(C)C=C2)C(=O)N(C)CC(=O)N[C@@H](CCC3=CC(F)=C(C(F)=C3)C(F)(F)F)C(=O)N4[C@@H](CCC4)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](C5CCCC5)C(=O)N(C)[C@H](C(=O)N(C)C)CC(=O)N(C)[C@@H](CC(C)C)C(=O)N1
""".strip()

fasta_path = Path("/tmp/example.fasta")
fasta_path.write_text(example_fasta)

output_dir = Path("/tmp/outputs")

candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    # 'default' setup
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
    device=torch.device("cuda:0"),
    use_esm_embeddings=True,
)

cif_paths = candidates.cif_paths
scores = [rd.aggregate_score for rd in candidates.ranking_data]


# Load pTM, ipTM, pLDDTs and clash scores for sample 2
scores = np.load(output_dir.joinpath("scores.model_idx_2.npz"))
