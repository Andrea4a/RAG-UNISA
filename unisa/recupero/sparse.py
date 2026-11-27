import re

import numpy as np
from rank_bm25 import BM25Okapi

from nltk.stem.snowball import ItalianStemmer
from langchain_core.documents import Document

from unisa.indicizzazione.embedding import crea_prefisso_contestuale

stemmer = ItalianStemmer()

STOP_WORDS = {
    "il", "lo", "la", "i", "gli", "le",
    "un", "uno", "una",
    "di", "a", "da", "in", "con", "su",
    "per", "tra", "fra",
    "l", "d",
    "che", "quale", "quali",
    "è", "sono"
}


def tokenizza_bm25(testo):

    testo = testo.lower()

    tokens = re.findall(r"\b\w+\b", testo)

    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in STOP_WORDS
    ]

    return tokens


def ricerca_bm25(domanda, bm25, ids, documenti, metadatas, k=5):

    query_tokenizzata = tokenizza_bm25(domanda)

    scores = bm25.get_scores(query_tokenizzata)

    top_indices = np.argsort(scores)[::-1]

    top_indices = [
        i for i in top_indices
        if any(
            token in bm25.doc_freqs[i]
            for token in query_tokenizzata
        )
    ][:k]

    risultati = []

    for i in top_indices:
        risultati.append({
            "child_id": ids[i],
            "testo": documenti[i],
            "metadata": metadatas[i],
            "score": float(scores[i])
        })

    return risultati

def crea_indice_bm25(collection):

    dati = collection.get(
        include=["documents", "metadatas"]
    )

    ids = dati["ids"]
    documenti = dati["documents"]
    metadatas = dati["metadatas"]

    corpus_tokenizzato = []

    for testo, metadata in zip(documenti, metadatas):

        child = Document(
            page_content=testo,
            metadata=metadata
        )

        prefisso = crea_prefisso_contestuale(child)

        testo_contestuale = f"{prefisso}\n\n{testo}"

        tokens = tokenizza_bm25(testo_contestuale)

        corpus_tokenizzato.append(tokens)

    bm25 = BM25Okapi(
        corpus_tokenizzato,
        k1=1.5,
        b=0.75
    )

    return bm25, ids, documenti, metadatas

def main():
    from unisa.indicizzazione.database import crea_client

    client = crea_client()

    collection = client.get_collection(
        name="unisa_baseline",
        embedding_function=None
    )

    bm25, ids, documenti, metadatas = crea_indice_bm25(collection)

    print("Numero child:", bm25.corpus_size)
    print("Lunghezza media:", bm25.avgdl)
    print("Dimensione vocabolario:", len(bm25.idf))

    domanda = "Quali sono le scadenze per presentare l'ISEE?"

    risultati = ricerca_bm25(
        domanda, bm25, ids, documenti, metadatas, k=5
    )

    for rank, risultato in enumerate(risultati, start=1):

        print(f"\n--- Rank {rank} ---")
        print("Child ID:", risultato["child_id"])
        print("Score:", risultato["score"])
        print("Testo:", risultato["testo"])

    query_tokenizzata = tokenizza_bm25(domanda)

    indice = ids.index("D05_P016_C001")

    for token in query_tokenizzata:

        idf = bm25.idf.get(token, 0)
        tf = bm25.doc_freqs[indice].get(token, 0)
        score = bm25.get_scores([token])[indice]

        print(
            f"{token:12} | "
            f"TF: {tf} | "
            f"IDF: {idf:.4f} | "
            f"Score: {score:.4f}"
        )


if __name__ == "__main__":
    main()
