"""Interfaccia interattiva da terminale per il RAG UniSA."""

import os
import time

from groq import APIError, Groq

from unisa.recupero.generazione import descrivi_fonte, genera_risposta
from unisa.recupero.pipeline import RetrievalPipeline


def main():
    chiave = os.getenv("GROQ_API_KEY", "").strip()
    client = Groq(api_key=chiave, timeout=30.0, max_retries=0) if chiave else None
    if client is None:
        print("GROQ_API_KEY assente. Modalità di ricerca locale attiva.")

    pipeline = RetrievalPipeline()
    domanda_pendente = None
    print("Digita 'exit' per uscire o 'reset' per iniziare una nuova domanda.")

    while True:
        try:
            domanda = input("\nDomanda: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if domanda.lower() == "exit":
            break
        if domanda.lower() == "reset":
            domanda_pendente = None
            print("Domanda precedente cancellata.")
            continue
        if not domanda:
            continue

        # Se il modello ha chiesto un chiarimento, conserviamo la domanda
        # precedente e aggiungiamo la nuova informazione fornita dall'utente.
        domanda_completa = domanda
        if domanda_pendente is not None:
            domanda_completa = f"{domanda_pendente}\nChiarimento utente: {domanda}"

        inizio_totale = time.perf_counter()
        risultati = pipeline.recupera_contesto(domanda_completa)
        latenza_retrieval = time.perf_counter() - inizio_totale

        if client is not None:
            try:
                tipo, risposta, fonti, statistiche = genera_risposta(
                    client, domanda_completa, risultati
                )
            except APIError as errore:
                stato = getattr(errore, "status_code", type(errore).__name__)
                print(f"Richiesta Groq non riuscita ({stato}).")
            except ValueError as errore:
                print(errore)
            else:
                latenza_totale = time.perf_counter() - inizio_totale
                print(f"\n{risposta}")

                for numero, fonte in fonti.items():
                    url = fonte["metadata"].get("url")
                    print(f"[{numero}] {descrivi_fonte(fonte)}")
                    if url:
                        print(f"    {url}")

                if tipo == "chiarimento":
                    domanda_pendente = (
                        f"{domanda_completa}\nDomanda di chiarimento: {risposta}"
                    )
                else:
                    domanda_pendente = None

                print("\n--- STATISTICHE ---")
                print("Tipo:", tipo)
                print(f"Latenza retrieval: {latenza_retrieval:.3f} s")
                print(f"Latenza LLM: {statistiche['latenza_llm']:.3f} s")
                print(f"Latenza totale: {latenza_totale:.3f} s")
                print("Input token:", statistiche["input_tokens"])
                print("Output token:", statistiche["output_tokens"])
                print("Token totali:", statistiche["total_tokens"])
                continue

            # In caso di errore, mostriamo il contesto senza inventare una risposta.
            print("\nMostro i brani recuperati senza generazione.")

        for i, parent in enumerate(risultati, start=1):
            print(f"\n--- Parent {i} ---")
            print("ID:", parent["parent_id"])
            print("Testo:", parent["testo"])

        print(f"\nLatenza retrieval: {latenza_retrieval:.3f} s")


if __name__ == "__main__":
    main()
