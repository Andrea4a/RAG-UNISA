"""Percorsi condivisi del progetto, indipendenti dalla directory di lancio."""

from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
DOCUMENTI = RADICE / "documenti"
CHROMA_DB = RADICE / "chroma_db"
REGOLE_PULIZIA = RADICE / "config" / "regole_pulizia.json"
INVENTARIO_EXCEL = DOCUMENTI / "inventario_documenti_unisa.xlsx"
DOMANDE = DOCUMENTI / "domande_di_prova_unisa.xlsx"
PARENTS_PATH = DOCUMENTI / "parents.json"
