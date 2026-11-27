import argparse
import subprocess
import sys


from unisa.percorsi import DOMANDE, INVENTARIO_EXCEL, RADICE


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepara i documenti UniSA fino alla produzione dei Markdown strutturati e di paragrafi.jsonl."
    )
    parser.add_argument(
        "--scarica",
        action="store_true",
        help="aggiorna le pagine web prima di elaborarle (di default usa i file locali)",
    )
    opzioni = parser.parse_args()

    for percorso in (INVENTARIO_EXCEL, DOMANDE):
        if not percorso.is_file():
            parser.error(f"File di input mancante: {percorso}")

    passi = [
        ["unisa.ingestione.raccolta", *([] if opzioni.scarica else ["--solo-locale"])],
        ["unisa.ingestione.estrazione_pdf"],
        ["unisa.ingestione.raccolta", "--solo-locale"],
        ["unisa.ingestione.pulizia"],
        ["unisa.ingestione.ricomposizione"],
        ["unisa.ingestione.struttura"],
    ]
    for passo in passi:
        comando = ["-m", *passo]
        descrizione = " ".join(comando)
        print(f"\n{'=' * 70}\n▶ python {descrizione}\n{'=' * 70}", flush=True)
        esito = subprocess.run([sys.executable, *comando], cwd=RADICE)
        if esito.returncode != 0:
            print(f"\n✗ Il passo «{descrizione}» si è fermato con un errore: la catena si interrompe qui.")
            return esito.returncode

    print("\n✓ Catena completata. Controlla in ogni passo la riga «Fatti persi» / «Fatti non trovati»: deve essere 0.")
    return 0
