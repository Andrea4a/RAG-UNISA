import chromadb
import numpy as np

from unisa.percorsi import CHROMA_DB as DATABASE_PATH


def crea_client():
    client = chromadb.PersistentClient(
        path=str(DATABASE_PATH)
    )

    return client

def crea_collection(client, nome: str):

    collection = client.get_or_create_collection(
        name=nome,
        embedding_function=None,
        configuration={
            "hnsw": {
                "space": "cosine",
                "max_neighbors": 16,
                "ef_construction": 100,
                "ef_search": 100
            }
        }
    )

    return collection


def prepara_dati(children, embeddings):

    if len(children) != len(embeddings):
        raise ValueError(
            "Numero di child ed embedding differente"
        )

    ids = []
    documents = []
    metadatas = []

    for child in children:

        ids.append(child.metadata["child_id"])

        documents.append(child.page_content)

        metadatas.append(prepara_metadata(child))

    return ids, documents, embeddings.tolist(), metadatas

def prepara_metadata(child) -> dict:

    campi = [
        "id",
        "child_id",
        "parent_id",
        "documento",
        "versione",
        "url",
        "H1",
        "H2",
        "H3",
        "start_index",
        "end_index",
        "numero_token",
        "pagine_child"
    ]

    metadata = {}

    for campo in campi:

        valore = child.metadata.get(campo)

        if valore is None or valore == "" or valore == []:
            continue

        metadata[campo] = valore

    return metadata

def inserisci_dati(collection, children, embeddings):

    ids, documents, vettori, metadatas = prepara_dati(
        children,
        embeddings
    )

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=vettori,
        metadatas=metadatas
    )

    print("Collection:", collection.name)
    print("Record presenti:", collection.count())


def verifica_collection(collection, children, embeddings):

    # 1. Mappatura degli ID agli indici originali
    child_map = {
        child.metadata["child_id"]: i
        for i, child in enumerate(children)
    }

    # 2. Recupero dei record da ChromaDB
    risultati = collection.get(
        include=[
            "documents",
            "embeddings",
            "metadatas"
        ]
    )

    # 3. Verifica degli identificativi
    ids_originali = set(child_map.keys())
    ids_database = set(risultati["ids"])

    if len(child_map) != len(children):
        raise ValueError("ID originali duplicati")

    if ids_originali != ids_database:
        raise ValueError("Gli ID non corrispondono")

    # 4. Contatori
    testi_coerenti = 0
    embedding_coerenti = 0
    metadata_coerenti = 0

    # 5. Confronto di ogni record
    for i, child_id in enumerate(risultati["ids"]):

        indice_originale = child_map[child_id]

        child_originale = children[indice_originale]
        embedding_originale = embeddings[indice_originale]

        # Verifica testo
        if risultati["documents"][i] == child_originale.page_content:
            testi_coerenti += 1

        # Verifica embedding
        if np.allclose(
            risultati["embeddings"][i],
            embedding_originale,
            atol=1e-6,
            rtol=0
        ):
            embedding_coerenti += 1

        # Verifica metadati
        if risultati["metadatas"][i] == prepara_metadata(child_originale):
            metadata_coerenti += 1

    # 6. Risultati
    totale = len(children)

    print("\n--- VERIFICA COLLECTION ---")
    print("Collection:", collection.name)
    print("Record attesi:", totale)
    print("Record recuperati:", len(risultati["ids"]))
    print("ID coerenti:", len(ids_originali))
    print("Testi coerenti:", testi_coerenti)
    print("Embedding coerenti:", embedding_coerenti)
    print("Metadati coerenti:", metadata_coerenti)

    if not (
        len(risultati["ids"]) == totale
        and testi_coerenti == totale
        and embedding_coerenti == totale
        and metadata_coerenti == totale
    ):
        raise ValueError("Verifica collection fallita")
    
if __name__ == "__main__":

    client = crea_client()

    collection_base = crea_collection(
        client, "unisa_baseline"
    )

    collection_contestuale = crea_collection(
        client, "unisa_contestuale"
    )

    print("Numero collection:", client.count_collections())

    print("Baseline:", collection_base.count())
    print("Contestuale:", collection_contestuale.count())