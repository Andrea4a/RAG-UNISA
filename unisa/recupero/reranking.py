from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from unisa.indicizzazione.embedding import crea_prefisso_contestuale

RERANKER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

def crea_testo_reranker(risultato, contestuale=False):

    testo = risultato["testo"]

    if contestuale:

        child = Document(
            page_content=testo,
            metadata=risultato["metadata"]
        )

        prefisso = crea_prefisso_contestuale(child)

        testo = f"{prefisso}\n\n{testo}"

    return testo

def carica_reranker():

    modello = CrossEncoder(
        RERANKER_NAME,
        max_length=512
    )

    return modello

def reranking(
    domanda,
    risultati_hybrid,
    modello,
    top_k=5,
    contestuale=False
):

    if not risultati_hybrid:
        return []

    coppie = [
        (
            domanda,
            crea_testo_reranker(risultato, contestuale)
        )
        for risultato in risultati_hybrid
    ]

    lunghezze = [
        len(
            modello.tokenizer.encode(
                query,
                text_pair=testo,
                add_special_tokens=True,
                truncation=False
            )
        )
        for query, testo in coppie
    ]

    print("Lunghezza minima:", min(lunghezze))
    print("Lunghezza massima:", max(lunghezze))
    print("Lunghezza media:", sum(lunghezze) / len(lunghezze))
    print("Input oltre 512 token:", sum(n > 512 for n in lunghezze))

    scores = modello.predict(
        coppie,
        batch_size=16
    )

    output = []

    for rank, (risultato, score) in enumerate(
        zip(risultati_hybrid, scores), start=1
    ):

        record = risultato.copy()

        record["rank_hybrid"] = rank
        record["score_reranker"] = float(score)

        output.append(record)

    ranking_finale = sorted(
        output,
        key=lambda x: x["score_reranker"],
        reverse=True
    )

    return ranking_finale[:top_k]

def main():

    modello = carica_reranker()

    domanda = "Quali sono le scadenze per presentare l'ISEE?"

    coppie = [
        (domanda, "L'ISEE va richiesto entro il 15 ottobre 2026."),
        (domanda, "L'ISEE può contenere omissioni e difformità."),
        (domanda, "Lo studente può rinunciare agli studi.")
    ]

    scores = modello.predict(coppie)

    print(scores)


if __name__ == "__main__":
    main()