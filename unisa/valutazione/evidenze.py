"""Mapping delle evidenze sui child e copertura del contesto UniSA.
Uso: python -m unisa.valutazione.evidenze
Non carica modelli e non modifica ChromaDB.
"""

import json
from unisa.percorsi import DOCUMENTI, PARENTS_PATH
GROUND_TRUTH_PATH = DOCUMENTI / "ground_truth.json"
MAPPING_PATH = DOCUMENTI / "ground_truth_mapping.json"
COLLECTIONS = ("unisa_baseline", "unisa_contestuale")


def carica_json(percorso):
    with percorso.open("r", encoding="utf-8") as file:
        return json.load(file)


def calcola_intersezione(inizio_a, fine_a, inizio_b, fine_b):
    return max(0, min(fine_a, fine_b) - max(inizio_a, inizio_b))


def unisci_intervalli(intervalli):
    if not intervalli:
        return []

    intervalli = sorted(intervalli)
    uniti = [intervalli[0]]

    for inizio, fine in intervalli[1:]:
        ultimo_inizio, ultima_fine = uniti[-1]
        if inizio <= ultima_fine:
            uniti[-1] = (ultimo_inizio, max(ultima_fine, fine))
        else:
            uniti.append((inizio, fine))

    return uniti


def raggruppa_intervalli(risultati):
    intervalli = {}
    for risultato in risultati:
        meta = risultato["metadata"]
        intervalli.setdefault(meta["parent_id"], []).append(
            (meta["start_index"], meta["end_index"])
        )

    return {pid: unisci_intervalli(righe)
            for pid, righe in intervalli.items()}


def raggruppa_intervalli_parents(parents):
    return {p["parent_id"]: [(0, len(p["testo"]))] for p in parents}


def copertura_evidenza(evidenza, intervalli, ignora_spazi=False):
    """Copertura dell'evidenza da parte dell'unione degli intervalli.
    Gli offset sono sempre in caratteri Python, con fine esclusiva.
    """
    pid = evidenza["parent_id"]
    inizio = evidenza["start_index"]
    fine = evidenza["end_index"]
    if not 0 <= inizio < fine:
        raise ValueError(f"Intervallo evidenza non valido: {pid}")

    uniti = unisci_intervalli(intervalli.get(pid, []))
    if not ignora_spazi:
        coperti = sum(calcola_intersezione(inizio, fine, a, b)
                      for a, b in uniti)
        return coperti / (fine - inizio)

    testo = evidenza["testo"]
    if len(testo) != fine - inizio:
        raise ValueError(f"Lunghezza evidenza non coerente: {pid}")

    totale = sum(not c.isspace() for c in testo)
    if not totale:
        raise ValueError(f"Evidenza priva di caratteri significativi: {pid}")

    coperti = 0
    for a, b in uniti:
        sinistra = max(inizio, a)
        destra = min(fine, b)
        if sinistra < destra:
            coperti += sum(not c.isspace()
                           for c in testo[sinistra-inizio:destra-inizio])
    return coperti / totale


def calcola_copertura(child, evidenza, ignora_spazi=False):
    return copertura_evidenza(
        evidenza, raggruppa_intervalli([child]), ignora_spazi
    )


def copertura_fatto(fatto, intervalli, ignora_spazi=True):
    # OR: basta una delle evidenze alternative.
    return max(copertura_evidenza(e, intervalli, ignora_spazi)
               for e in fatto["evidenze_alternative"])


def valuta_copertura_domanda(domanda, intervalli, ignora_spazi=True):
    """Macro-media delle coperture dei fatti e strict fact recall.
    Non si valutano qui l'astensione o le domande ambigue.
    """
    if domanda["answerable"] is not True:
        return None

    coperture = [copertura_fatto(f, intervalli, ignora_spazi)
                 for f in domanda["fatti"]]
    if not coperture:
        raise ValueError(f"Domanda senza fatti: {domanda['query_id']}")

    completi = sum(c == 1.0 for c in coperture)
    return {
        "context_recall": sum(coperture) / len(coperture),
        "strict_fact_recall": completi / len(coperture),
        "fatti_completi": completi,
        "fatti_totali": len(coperture),
        "copertura_fatti": {
            f["fact_id"]: c for f, c in zip(domanda["fatti"], coperture)
        }
    }


def verifica_ground_truth(ground_truth, archivio):
    ids = set()
    n_fatti = n_evidenze = 0
    for domanda in ground_truth:
        qid = domanda["query_id"]
        if qid in ids or domanda["set"] != "prova":
            raise ValueError(f"ID duplicato o set non ammesso: {qid}")
        ids.add(qid)
        fatti = domanda["fatti"]
        if (domanda["answerable"] is True) != bool(fatti):
            raise ValueError(f"Fatti incoerenti con answerable: {qid}")

        fact_ids = set()
        for fatto in fatti:
            fid = fatto["fact_id"]
            if fid in fact_ids or not fatto["evidenze_alternative"]:
                raise ValueError(f"Fatto duplicato o senza evidenze: {qid}/{fid}")
            fact_ids.add(fid)
            n_fatti += 1
            for e in fatto["evidenze_alternative"]:
                pid = e["parent_id"]
                testo = archivio[pid]["testo"]
                a, b = e["start_index"], e["end_index"]
                if not (0 <= a < b <= len(testo)):
                    raise ValueError(f"Offset non valido: {qid}/{fid}/{pid}")
                if testo[a:b] != e["testo"] or not e["testo"].strip():
                    raise ValueError(f"Evidenza incoerente: {qid}/{fid}/{pid}")
                n_evidenze += 1

    print(f"Ground truth verificato: {len(ids)} domande, "
          f"{n_fatti} fatti, {n_evidenze} evidenze")


def carica_children(collection, archivio):
    dati = collection.get(include=["documents", "metadatas"])
    children = []
    ids = set()
    for cid, testo, meta in zip(dati["ids"], dati["documents"], dati["metadatas"]):
        if cid in ids or meta["child_id"] != cid:
            raise ValueError(f"ID child incoerente: {cid}")
        ids.add(cid)
        parent = archivio[meta["parent_id"]]["testo"]
        a, b = meta["start_index"], meta["end_index"]
        if not (0 <= a < b <= len(parent)) or parent[a:b] != testo:
            raise ValueError(f"Child non coerente con il parent: {cid}")
        children.append({"child_id": cid, "testo": testo, "metadata": meta})

    if not children or len(children) != collection.count():
        raise ValueError("Numero di child non coerente o collection vuota")
    return sorted(children, key=lambda c: c["child_id"])


def crea_mapping(ground_truth, children):
    """Per ogni evidenza, elenca i child con sovrapposizione significativa."""
    per_parent = {}
    for child in children:
        per_parent.setdefault(child["metadata"]["parent_id"], []).append(child)

    mapping = []
    for domanda in ground_truth:
        record = {
            "query_id": domanda["query_id"],
            "domanda": domanda["domanda"],
            "answerable": domanda["answerable"],
            "fatti": []
        }
        for fatto in domanda["fatti"]:
            f = {"fact_id": fatto["fact_id"],
                 "descrizione": fatto["descrizione"],
                 "evidenze_alternative": []}
            for evidenza in fatto["evidenze_alternative"]:
                candidati = []
                for child in per_parent.get(evidenza["parent_id"], []):
                    copertura = calcola_copertura(child, evidenza, ignora_spazi=True)
                    if copertura > 0:
                        candidati.append({
                            "child_id": child["child_id"],
                            "copertura_evidenza": copertura,
                            "tipo": "completa" if copertura == 1.0 else "parziale"
                        })
                candidati.sort(key=lambda c: (-c["copertura_evidenza"], c["child_id"]))
                f["evidenze_alternative"].append({
                    **evidenza, "children": candidati
                })
            record["fatti"].append(f)
        mapping.append(record)
    return mapping


def verifica_copertura_oracle(ground_truth, children):
    """Limite superiore con TUTTI i child, non risultati di retrieval."""
    intervalli = raggruppa_intervalli(children)
    recall = []
    strict = []
    totali = completi_singolo = completi_unione = 0

    per_parent = {}
    for c in children:
        per_parent.setdefault(c["metadata"]["parent_id"], []).append(c)

    for domanda in ground_truth:
        metriche = valuta_copertura_domanda(domanda, intervalli)
        if metriche is None:
            continue
        recall.append(metriche["context_recall"])
        strict.append(metriche["strict_fact_recall"])
        for fatto in domanda["fatti"]:
            for e in fatto["evidenze_alternative"]:
                totali += 1
                c = copertura_evidenza(e, intervalli, ignora_spazi=True)
                if any(calcola_copertura(ch, e, True) == 1.0
                       for ch in per_parent.get(e["parent_id"], [])):
                    completi_singolo += 1
                if c == 1.0:
                    completi_unione += 1
                else:
                    print(f"Evidenza incompleta: {domanda['query_id']} / "
                          f"{fatto['fact_id']} / {e['parent_id']}: {c:.2%}")

    print("\n--- COPERTURA ORACLE (tutti i child) ---")
    print("Domande valutabili:", len(recall))
    print("Evidenze totali:", totali)
    print("Evidenze complete in almeno un child:", completi_singolo)
    print("Evidenze complete solo tramite unione:", completi_unione - completi_singolo)
    print("Evidenze incomplete anche con tutti i child:", totali - completi_unione)
    if recall:
        print(f"Context Recall (macro): {sum(recall)/len(recall):.4f}")
        print(f"Strict Fact Recall (macro): {sum(strict)/len(strict):.4f}")


def test_funzioni():
    assert calcola_intersezione(100, 200, 50, 150) == 50
    assert calcola_intersezione(100, 200, 250, 300) == 0
    assert unisci_intervalli([(100, 200), (150, 250), (300, 400)]) == [(100, 250), (300, 400)]
    assert unisci_intervalli([(100, 300), (150, 200)]) == [(100, 300)]
    assert unisci_intervalli([(0, 5), (5, 10)]) == [(0, 10)]
    e = {"parent_id": "P001", "start_index": 100, "end_index": 200}
    assert copertura_evidenza(e, {"P001": [(50, 130), (150, 250)]}) == 0.8
    assert copertura_evidenza(e, {"P002": [(100, 200)]}) == 0.0
    e = {"parent_id": "P001", "start_index": 0, "end_index": 5, "testo": "ab cd"}
    assert copertura_evidenza(e, {"P001": [(0, 2), (3, 5)]}, True) == 1.0
    assert copertura_evidenza(e, {"P001": [(0, 2), (3, 5)]}, False) == 0.8
    print("Test delle funzioni: OK")


def main():
    from unisa.indicizzazione.database import crea_client

    test_funzioni()
    archivio = carica_json(PARENTS_PATH)
    ground_truth = carica_json(GROUND_TRUTH_PATH)
    verifica_ground_truth(ground_truth, archivio)
    print("Parent caricati:", len(archivio))

    client = crea_client()
    children_base = None
    for nome in COLLECTIONS:
        collection = client.get_collection(name=nome, embedding_function=None)
        children = carica_children(collection, archivio)
        print(f"{nome}: {len(children)} child verificati")
        if children_base is None:
            children_base = children
        elif children != children_base:
            raise ValueError("Le collection non hanno gli stessi child/testi/metadati")

    print("Collection coerenti: OK")
    print("\n--- CREAZIONE MAPPING ---")
    mapping = crea_mapping(ground_truth, children_base)
    with MAPPING_PATH.open("w", encoding="utf-8") as file:
        json.dump(mapping, file, ensure_ascii=False, indent=2)
    print("Mapping salvato:", MAPPING_PATH)

    verifica_copertura_oracle(ground_truth, children_base)


if __name__ == "__main__":
    main()
