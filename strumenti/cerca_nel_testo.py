"""Mostra le occorrenze dei termini di controllo nei testi estratti."""

import re
from pathlib import Path


TESTI = Path(__file__).resolve().parents[1] / "documenti" / "testo"
TERMINI = ("pergamena", "denominazione", "password", "credenziali", "autonom", "6.500", "6500")


def main() -> None:
    for file in sorted(TESTI.glob("*.md")):
        testo = file.read_text(encoding="utf-8")
        for termine in TERMINI:
            for occorrenza in re.finditer(re.escape(termine), testo, re.IGNORECASE):
                inizio = max(0, occorrenza.start() - 150)
                fine = occorrenza.end() + 150
                contesto = testo[inizio:fine].replace("\n", " ")
                print(f"{file.stem}  {termine!r}: ...{contesto}...\n")


if __name__ == "__main__":
    main()
