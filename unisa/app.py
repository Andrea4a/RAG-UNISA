"""Punto di avvio compatibile con il percorso storico."""

if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unisa.interfacce.web import main


if __name__ == "__main__":
    main()
