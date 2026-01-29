
import csv
import sys
from pathlib import Path
from dataset_utils import get_domain_targets_with_counts

if __name__ == "__main__":
    results = get_domain_targets_with_counts("PDZ", 4, 512)
    
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "targets_pdz.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Accession", "Name", "Binding_Sequences_Count"])
        for count, accession, name in results:
            writer.writerow([accession, name, count])
            
    print(f"Written {len(results)} rows to {output_path}", file=sys.stderr)
