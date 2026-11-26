#!/bin/bash

set -e  # stoppe au premier script qui échoue

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$ROOT/scripts"
LOGS="$ROOT/logs"

mkdir -p "$LOGS"

LOGFILE="$LOGS/pipeline_$(date +%Y%m%d_%H%M%S).txt"

exec > >(tee -a "$LOGFILE") 2>&1

echo "Pipeline sh lancé le $(date)"

SECONDS=0

formater_duree() {
    printf "%dh %02dmin %02ds" $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60))
}

# affiché aussi en cas d'échec d'une étape (set -e)
trap 'echo "Durée totale : $(formater_duree $SECONDS)"' EXIT

etapes=(
    "1_1_extraction.py"
    "1_2_fusion_legislatures.py"
    "1_3_regroupement_interventions.py"
    "2_1_filtrage.py"
    "2_2_identification_acteurs.py"
    "2_3_match_deputes.py"
    "2_4_affiliations.py"
    "2_5_commissions.py"
    "3_1_identification_republique.py"
    # TODO : intégrer step 4 stat
    "5_1_embeddings.py"
    "5_2_topic_modeling"
)

cd "$SCRIPTS"

for etape in "${etapes[@]}"; do
    echo
    echo "========================================"
    echo "=== $etape ==="
    echo "========================================"

    debut_etape=$SECONDS
    python -u "$etape"
    echo "--- $etape : $(formater_duree $((SECONDS - debut_etape)))"
done

echo
echo "========================================"
echo "=== Pipeline .sh terminé ! ==="
echo "========================================"

echo "Journal : $LOGFILE"

# to run :
# chmod +x run_pipeline.sh
# ./run_pipeline.sh