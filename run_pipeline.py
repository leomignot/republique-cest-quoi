# TODO : sans doute passer à un vrai log pour conserver infos ?

import subprocess
import sys
from pathlib import Path

scripts = Path(__file__).parent / "scripts"

etapes = [
    "1_1_extraction.py",
    "1_2_fusion_legislatures.py",
    "1_3_regroupement_interventions.py",
    "2_1_filtrage.py",
    "2_2_identification_acteurs.py",
    "2_3_match_deputes.py",
    "2_4_affiliations.py",
    "2_5_commissions.py",
    "3_1_identification_republique.py",
    # TODO : intégrer step 4 stat
    "5_1_embeddings.py",
    "5_2_topic_modeling",
]

for etape in etapes:
    print(f"\n{'=' * 40}\n=== {etape} ===\n{'=' * 40}")
    subprocess.run(
        [sys.executable, etape],
        cwd=scripts,  # exécuter depuis scripts/ pour que ../data/ fonctionne
        check=True,
    )

print(f"\n{'=' * 40}\n=== Pipeline py complet ! ===\n{'=' * 40}")

# pour exécuter :
# python run_pipeline.py
