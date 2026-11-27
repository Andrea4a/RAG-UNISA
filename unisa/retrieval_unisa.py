"""Punto di avvio compatibile con il percorso storico."""

if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unisa.interfacce.terminale import main
from unisa.recupero.generazione import MODELLO_GROQ, SYSTEM_PROMPT, descrivi_fonte, genera_risposta
from unisa.recupero.pipeline import TOP_K_FINALE, RetrievalPipeline


if __name__ == "__main__":
    main()
