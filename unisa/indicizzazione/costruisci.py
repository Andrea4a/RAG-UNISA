"""Costruisce e verifica l'indice ChromaDB a partire dal corpus strutturato."""

from langchain_core.documents import Document

from unisa.indicizzazione.archivio import salva_parents
from unisa.indicizzazione.chunking import (
    CARTELLA_DOCUMENTI, INVENTARIO_PATH, carica_inventario, carica_righe_tabelle,
    crea_children, crea_parents, individua_righe_tabelle, verifica_parents,
)
from unisa.indicizzazione.database import crea_client, crea_collection, inserisci_dati, verifica_collection
from unisa.indicizzazione.diagnostica import (
    analizza_children, analizza_overlap, analizza_token, stampa_statistiche,
    verifica_copertura, verifica_integrita_tabelle,
)
from unisa.indicizzazione.embedding import (
    carica_modello, genera_embedding, verifica_embedding, verifica_input_embedding,
)
from unisa.percorsi import DOCUMENTI

PARAGRAFI_PATH = DOCUMENTI / "paragrafi.jsonl"


def main() -> tuple[list[Document], list[Document]]:
    """Esegue il chunking, le verifiche e l’analisi delle righe di tabella."""
    print("--- CREAZIONE DEI PARENT ---\n")

    inventario = carica_inventario(INVENTARIO_PATH)

    all_parents = crea_parents(CARTELLA_DOCUMENTI, inventario)

    verifica_parents(all_parents)

    salva_parents(all_parents)

    stampa_statistiche(all_parents)

    model = carica_modello()
    tokenizer = model.tokenizer

    analizza_token(all_parents, tokenizer)

    all_children = crea_children(all_parents, tokenizer)

    print("\n--- CREAZIONE CHILD ---")
    print("Parent totali:", len(all_parents))
    print("Child totali:", len(all_children))

    analizza_children(all_children)

    analizza_overlap(all_parents, all_children, tokenizer)

    verifica_copertura(all_parents, all_children)

    righe_tabelle = carica_righe_tabelle(PARAGRAFI_PATH)

    righe_individuate = individua_righe_tabelle(all_parents, righe_tabelle)

    verifica_integrita_tabelle(
        righe_individuate,
        all_children
    )

    verifica_input_embedding(
        all_children,
        tokenizer,
        contestuale=False
    )

    verifica_input_embedding(
        all_children,
        tokenizer,
        contestuale=True
    )

    embeddings_base = genera_embedding(
        all_children,
        model,
        contestuale=False
    )

    embeddings_contestuali = genera_embedding(
        all_children,
        model,
        contestuale=True
    )

    print("Shape baseline:", embeddings_base.shape)
    print("Shape contestuale:", embeddings_contestuali.shape)

    verifica_embedding(
        embeddings_base,
        len(all_children)
    )

    verifica_embedding(
        embeddings_contestuali,
        len(all_children)
    )

    client = crea_client()

    collection_base = crea_collection(
        client, "unisa_baseline"
    )

    collection_contestuale = crea_collection(
        client, "unisa_contestuale"
    )

    inserisci_dati(
        collection_base,
        all_children,
        embeddings_base
    )

    inserisci_dati(
        collection_contestuale,
        all_children,
        embeddings_contestuali
    )

    verifica_collection(
        collection_base,
        all_children,
        embeddings_base
    )

    verifica_collection(
        collection_contestuale,
        all_children,
        embeddings_contestuali
    )

    return all_parents, all_children


if __name__ == "__main__":
    main()
