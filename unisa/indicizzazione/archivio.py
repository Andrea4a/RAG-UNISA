import json
from unisa.percorsi import PARENTS_PATH


def salva_parents(parents):

    archivio = {}

    for parent in parents:

        parent_id = parent.metadata["parent_id"]

        archivio[parent_id] = {
            "testo": parent.page_content,
            "metadata": parent.metadata
        }

    with open(PARENTS_PATH, "w", encoding="utf-8") as file:

        json.dump(
            archivio,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("Parent salvati:", len(archivio))

def carica_parents():

    with open(PARENTS_PATH, "r", encoding="utf-8") as file:

        archivio = json.load(file)

    return archivio

def verifica_collegamento_parents(collection, archivio):

    risultati = collection.get(
        include=["documents", "metadatas"]
    )

    coerenti = 0

    for testo_child, metadata in zip(
        risultati["documents"],
        risultati["metadatas"]
    ):

        parent_id = metadata["parent_id"]

        if parent_id not in archivio:
            raise ValueError(f"Parent mancante: {parent_id}")

        testo_parent = archivio[parent_id]["testo"]

        inizio = metadata["start_index"]
        fine = metadata["end_index"]

        if testo_parent[inizio:fine] != testo_child:
            raise ValueError(
                f"Testo non coerente: {metadata['child_id']}"
            )

        coerenti += 1

    print("Child verificati:", coerenti)

def espandi_parents(risultati, archivio):

    parents_recuperati = {}

    for rank, risultato in enumerate(risultati, start=1):

        parent_id = risultato["metadata"]["parent_id"]
        child_id = risultato["child_id"]

        if parent_id not in parents_recuperati:

            parent = archivio[parent_id]

            parents_recuperati[parent_id] = {
                "parent_id": parent_id,
                "testo": parent["testo"],
                "metadata": parent["metadata"],
                "rank_primo_child": rank,
                "children_recuperati": []
            }

        parents_recuperati[parent_id]["children_recuperati"].append(
            risultato
        )

    return list(parents_recuperati.values())

if __name__ == "__main__":

    from unisa.indicizzazione.database import crea_client

    client = crea_client()

    archivio = carica_parents()

    for nome in ["unisa_baseline", "unisa_contestuale"]:

        collection = client.get_collection(
            name=nome,
            embedding_function=None
        )

        print("\nCollection:", nome)
        print("Parent caricati:", len(archivio))

        verifica_collegamento_parents(collection, archivio)