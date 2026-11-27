from unisa.indicizzazione.database import crea_client
from unisa.indicizzazione.embedding import carica_modello, genera_embedding_query

def ricerca_dense(collection, embedding_query, k=5):

    risultati = collection.query(
        query_embeddings=[embedding_query.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    output = []

    for i, child_id in enumerate(risultati["ids"][0]):

        output.append({
            "child_id": child_id,
            "testo": risultati["documents"][0][i],
            "metadata": risultati["metadatas"][0][i],
            "score": 1 - risultati["distances"][0][i]
        })

    return output

def main():

    client = crea_client()

    collection = client.get_collection(
        name="unisa_baseline",
        embedding_function=None
    )

    modello = carica_modello()

    domanda = "Quali sono le scadenze per presentare l'ISEE?"

    embedding_query = genera_embedding_query(domanda, modello)

    risultati = ricerca_dense(collection, embedding_query, k=5)

    print(risultati)


if __name__ == "__main__":
    main()