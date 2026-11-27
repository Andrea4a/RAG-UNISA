from unisa.recupero.dense import ricerca_dense
from unisa.recupero.sparse import ricerca_bm25
from unisa.indicizzazione.embedding import genera_embedding_query
from unisa.indicizzazione.database import crea_client
from unisa.indicizzazione.embedding import carica_modello
from unisa.recupero.sparse import crea_indice_bm25
from unisa.recupero.reranking import carica_reranker, reranking
from unisa.indicizzazione.archivio import carica_parents, espandi_parents

def ricerca_hybrid(
    domanda,
    modello,
    collection,
    bm25,
    ids,
    documenti,
    metadatas,
    k_dense=20,
    k_bm25=20,
    k0=60,
    top_k=5
):

    embedding_query = genera_embedding_query(domanda, modello)

    risultati_dense = ricerca_dense(
        collection,
        embedding_query,
        k=k_dense
    )

    risultati_bm25 = ricerca_bm25(
        domanda,
        bm25,
        ids,
        documenti,
        metadatas,
        k=k_bm25
    )

    risultati_hybrid = reciprocal_rank_fusion(
        risultati_dense,
        risultati_bm25,
        k0=k0,
        top_k=top_k
    )

    return risultati_hybrid

def reciprocal_rank_fusion(
    risultati_dense,
    risultati_bm25,
    k0=60,
    top_k=5,
    w_dense=0.7,
    w_bm25=0.3
):

    scores_rrf = {}
    informazioni = {}

    for metodo, ranking, peso in [
        ("dense", risultati_dense, w_dense),
        ("bm25", risultati_bm25, w_bm25)
    ]:

        for rank, risultato in enumerate(ranking, start=1):

            child_id = risultato["child_id"]

            if child_id not in scores_rrf:

                scores_rrf[child_id] = 0.0

                informazioni[child_id] = {
                    "child_id": child_id,
                    "testo": risultato["testo"],
                    "metadata": risultato["metadata"],
                    "rank_dense": None,
                    "rank_bm25": None,
                    "score_dense": None,
                    "score_bm25": None
                }

            scores_rrf[child_id] += peso / (k0 + rank)

            informazioni[child_id][f"rank_{metodo}"] = rank
            informazioni[child_id][f"score_{metodo}"] = risultato["score"]

    ranking_finale = sorted(
        scores_rrf.items(),
        key=lambda x: x[1],
        reverse=True
    )

    output = []

    for child_id, score in ranking_finale[:top_k]:

        risultato = informazioni[child_id]
        risultato["score_rrf"] = score

        output.append(risultato)

    return output

def main():

    client = crea_client()

    collection = client.get_collection(
        name="unisa_baseline",
        embedding_function=None
    )

    modello = carica_modello()
    modello_reranker = carica_reranker()
    
    bm25, ids, documenti, metadatas = crea_indice_bm25(collection)

    domanda = "Quali sono le scadenze per presentare l'ISEE?"

    risultati_hybrid = ricerca_hybrid(
        domanda,
        modello,
        collection,
        bm25,
        ids,
        documenti,
        metadatas,
        k_dense=40,
        k_bm25=40,
        k0=60,
        top_k=20
    )

    risultati_finali = reranking(
        domanda,
        risultati_hybrid,
        modello_reranker,
        top_k=5,
        contestuale=True
    )

    archivio = carica_parents()

    parents_finali = espandi_parents(
        risultati_finali,
        archivio
    )

    print("\nChild recuperati:", len(risultati_finali))
    print("Parent distinti:", len(parents_finali))

    for rank, parent in enumerate(parents_finali, start=1):

        print(f"\n--- Parent {rank} ---")
        print("Parent ID:", parent["parent_id"])
        print("Rank primo child:", parent["rank_primo_child"])

        print("Child associati:", [
            child["child_id"]
            for child in parent["children_recuperati"]
        ])

        print("Testo:", parent["testo"])
        
    for rank, risultato in enumerate(risultati_finali, start=1):

        print(f"\n--- Rank Reranker {rank} ---")
        print("Child ID:", risultato["child_id"])

        print("Rank Dense:", risultato["rank_dense"])
        print("Rank BM25:", risultato["rank_bm25"])
        print("Rank Hybrid:", risultato["rank_hybrid"])

        print("Score Dense:", risultato["score_dense"])
        print("Score BM25:", risultato["score_bm25"])
        print("Score RRF:", risultato["score_rrf"])
        print("Score Reranker:", risultato["score_reranker"])

        print("Testo:", risultato["testo"])


if __name__ == "__main__":
    main()