import os
import sys
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Regex to parse the canonical Mendeley filename convention:
#   {COLLECTION}-{SUBJECT}-{SESSION}-HUMAN.csv
#   {COLLECTION}-{SUBJECT}-{SESSION}-{Category}-{Synthesizer}.csv
FILENAME_RE = re.compile(
    r"^(?P<collection>[A-Z]+)-"          # e.g. KM, GAY, GUN, LSIA, REVIEW
    r"(?P<subject>.+?)-"                 # e.g. s019, A10075453GKWMB0SRB9SQ
    r"(?P<session>\d+)-"                 # e.g. 1, 2, 3, 4
    r"(?P<remainder>.+)\.csv$",          # HUMAN  or  BetweenSubject-AverageSynthesizer
    re.IGNORECASE,
)

# Known synthesiser suffixes (used for classification)
SYNTHESIZER_NAMES = {
    "AverageSynthesizer",
    "GaussianSynthesizer",
    "HistogramSynthesizer",
    "NonStationaryHistogramSynthesizer",
    "UniformSynthesizer",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CSVRecord:
    """Metadata for a single CSV file in the Mendeley dataset."""
    filepath: Path
    collection: str       # GAY | GUN | KM | LSIA | REVIEW
    subject: str          # unique subject identifier
    session: int          # session number
    is_human: bool        # True if HUMAN sample
    synth_category: str   # BetweenSubject / WithinSubject100 / … (empty for human)
    synth_type: str       # AverageSynthesizer / … (empty for human)

    @property
    def subject_uid(self) -> str:
        """Globally unique subject key = collection + subject."""
        return f"{self.collection}_{self.subject}"


@dataclass
class DataManifest:
    """Complete catalogue of every CSV in the raw dataset."""
    human_records: List[CSVRecord] = field(default_factory=list)
    synth_records: List[CSVRecord] = field(default_factory=list)

    # Quick look-ups built after scanning
    collections: set = field(default_factory=set)
    unique_subjects: set = field(default_factory=set)
    synth_types_found: set = field(default_factory=set)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "DATA MANIFEST SUMMARY",
            "=" * 60,
            f"  Collections found   : {sorted(self.collections)}",
            f"  Unique subjects     : {len(self.unique_subjects)}",
            f"  Human CSV files     : {len(self.human_records):,}",
            f"  Synthesized CSV files: {len(self.synth_records):,}",
            f"  Synthesizer types   : {sorted(self.synth_types_found)}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core scanning logic
# ---------------------------------------------------------------------------
def _parse_filename(filepath: Path) -> CSVRecord | None:
    """Parse a single CSV filename into a structured record."""
    name = filepath.name
    m = FILENAME_RE.match(name)
    if m is None:
        print(f"DEBUG: Skipping non-matching file: {name}")
        return None

    collection = m.group("collection").upper()
    subject = m.group("subject")
    session = int(m.group("session"))
    remainder = m.group("remainder")

    if remainder.upper() == "HUMAN":
        return CSVRecord(
            filepath=filepath,
            collection=collection,
            subject=subject,
            session=session,
            is_human=True,
            synth_category="",
            synth_type="",
        )

    # Try to split remainder into category-SynthesizerName
    for sname in SYNTHESIZER_NAMES:
        if remainder.endswith(sname):
            category = remainder[: -(len(sname) + 1)]  # strip trailing -SynthName
            return CSVRecord(
                filepath=filepath,
                collection=collection,
                subject=subject,
                session=session,
                is_human=False,
                synth_category=category,
                synth_type=sname,
            )

    print(f"WARNING: Could not classify remainder='{remainder}' for {name}")
    return None


def scan_raw_data(raw_dir: Path = RAW_DATA_DIR) -> DataManifest:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        print(
            f"FATAL: Mendeley dataset CSVs missing. "
            f"Expected directory does not exist: {raw_dir}"
        )
        sys.exit(1)

    csv_files = sorted(raw_dir.rglob("*.csv"))

    if len(csv_files) == 0:
        print(
            f"FATAL: Mendeley dataset CSVs missing. "
            f"No .csv files found under {raw_dir}"
        )
        sys.exit(1)

    print(f"INFO: Found {len(csv_files):,} CSV files under {raw_dir}")
    manifest = DataManifest()
    skipped = 0

    for csv_path in csv_files:
        record = _parse_filename(csv_path)
        if record is None:
            skipped += 1
            continue

        manifest.collections.add(record.collection)
        manifest.unique_subjects.add(record.subject_uid)

        if record.is_human:
            manifest.human_records.append(record)
        else:
            manifest.synth_records.append(record)
            manifest.synth_types_found.add(record.synth_type)

    if skipped > 0:
        print(f"WARNING: Skipped {skipped} files that did not match naming convention.")

    if len(manifest.human_records) == 0:
        print(
            "FATAL: Mendeley dataset CSVs missing. "
            "Scanned directory contained CSV files but none matched the "
            "expected HUMAN filename pattern."
        )
        sys.exit(1)

    print(manifest.summary())
    return manifest


# ---------------------------------------------------------------------------
# Convenience helpers for downstream modules
# ---------------------------------------------------------------------------
def get_subjects_by_collection(manifest: DataManifest) -> Dict[str, List[str]]:
    """Return {collection: [sorted list of subject UIDs]}."""
    coll_map: Dict[str, set] = {}
    for rec in manifest.human_records:
        coll_map.setdefault(rec.collection, set()).add(rec.subject_uid)
    return {k: sorted(v) for k, v in sorted(coll_map.items())}


def get_synth_records_for_subject(
    manifest: DataManifest, subject_uid: str
) -> List[CSVRecord]:
    """Return all synthesized CSVRecords belonging to a given subject."""
    return [r for r in manifest.synth_records if r.subject_uid == subject_uid]


# ---------------------------------------------------------------------------
# Stand-alone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("INFO: Phase 1: Data Acquisition & Validation")
    print(f"INFO: Scanning raw data directory: {RAW_DATA_DIR}")

    manifest = scan_raw_data()

    # Print per-collection breakdown
    by_coll = get_subjects_by_collection(manifest)
    for coll, subjects in by_coll.items():
        n_human = sum(1 for r in manifest.human_records if r.collection == coll)
        n_synth = sum(1 for r in manifest.synth_records if r.collection == coll)
        print(
            f"INFO:   [{coll}]  subjects={len(subjects):>4}  "
            f"human_files={n_human:>6,}  synth_files={n_synth:>8,}"
        )

    print("INFO: Phase 1 COMPLETE — data validated successfully.")
