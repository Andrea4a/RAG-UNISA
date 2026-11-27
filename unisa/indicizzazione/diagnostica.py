"""Controlli di copertura, token, overlap e integrità delle tabelle."""

import statistics

from langchain_core.documents import Document

from unisa.indicizzazione.chunking import CHUNK_SIZE
from unisa.indicizzazione.embedding import conta_token


def stampa_statistiche(parents: list[Document]) -> None:
    """Stampa le dimensioni dei parent in caratteri."""
    dimensioni = [
        len(parent.page_content)
        for parent in parents
    ]

    print("\n--- STATISTICHE PARENT ---")

    print(f"Parent totali:      {len(parents)}")
    print(f"Dimensione minima:  {min(dimensioni)} caratteri")
    print(f"Dimensione massima:  {max(dimensioni)} caratteri")
    print(f"Dimensione media:   {statistics.mean(dimensioni):.2f}")
    print(f"Mediana:            {statistics.median(dimensioni):.2f}")

    print("\n--- DISTRIBUZIONE DELLE DIMENSIONI ---")

    intervalli = [
        (0, 200),
        (200, 400),
        (400, 800),
        (800, 1600)
    ]

    for minimo, massimo in intervalli:
        numero = sum(minimo < d <= massimo for d in dimensioni)

        percentuale = numero / len(parents) * 100

        print(
            f"{minimo + 1:4}-{massimo:4}: "
            f"{numero:3} parent ({percentuale:.1f}%)"
        )

    oltre = sum(d > 1600 for d in dimensioni)

    print(
        f"Oltre 1600: "
        f"{oltre:3} parent "
        f"({oltre / len(parents) * 100:.1f}%)"
    )


def analizza_token(parents: list[Document], tokenizer) -> None:
    """Stampa le dimensioni dei parent in token."""
    dimensioni = [
        conta_token(parent.page_content, tokenizer)
        for parent in parents
    ]

    print("\n--- STATISTICHE IN TOKEN ---")

    print(f"Parent totali: {len(parents)}")
    print(f"Minimo:        {min(dimensioni)}")
    print(f"Massimo:       {max(dimensioni)}")
    print(f"Media:         {statistics.mean(dimensioni):.2f}")
    print(f"Mediana:       {statistics.median(dimensioni):.2f}")

    intervalli = [
        (0, 64),
        (64, 128),
        (128, 256),
        (256, 384),
        (384, 512)
    ]

    print("\n--- DISTRIBUZIONE IN TOKEN ---")

    for minimo, massimo in intervalli:
        numero = sum(minimo < d <= massimo for d in dimensioni)

        percentuale = numero / len(parents) * 100

        print(
            f"{minimo + 1:3}-{massimo:3}: "
            f"{numero:3} parent ({percentuale:.2f}%)"
        )

    oltre = sum(d > 512 for d in dimensioni)

    print(f"Oltre 512: {oltre} parent")


def analizza_children(children: list[Document]) -> None:
    """Stampa le dimensioni dei child e segnala i segmenti oltre il limite."""
    dimensioni = [
        child.metadata["numero_token"]
        for child in children
    ]

    print("\n--- STATISTICHE CHILD ---")

    print(f"Child totali: {len(children)}")
    print(f"Minimo:       {min(dimensioni)}")
    print(f"Massimo:      {max(dimensioni)}")
    print(f"Media:        {statistics.mean(dimensioni):.2f}")
    print(f"Mediana:      {statistics.median(dimensioni):.2f}")

    intervalli = [
        (0, 32),
        (32, 64),
        (64, 96),
        (96, 128)
    ]

    for minimo, massimo in intervalli:
        numero = sum(minimo < d <= massimo for d in dimensioni)

        print(f"{minimo + 1:3}-{massimo:3}: {numero} child")

    oltre_limite = [
        child for child in children
        if child.metadata["numero_token"] > CHUNK_SIZE
    ]

    print(f"Child oltre {CHUNK_SIZE} token: {len(oltre_limite)}")

    for child in oltre_limite:
        print(
            child.metadata["child_id"],
            child.metadata["numero_token"]
        )


def analizza_overlap(
    parents: list[Document],
    children: list[Document],
    tokenizer
) -> None:
    """Misura la sovrapposizione tra child consecutivi dello stesso parent."""
    parent_map = {
        p.metadata["parent_id"]: p
        for p in parents
    }

    overlaps = []

    for precedente, corrente in zip(children, children[1:]):
        if precedente.metadata["parent_id"] != corrente.metadata["parent_id"]:
            continue

        parent = parent_map[corrente.metadata["parent_id"]]

        inizio = corrente.metadata["start_index"]
        fine = precedente.metadata["end_index"]

        if inizio < fine:
            testo_condiviso = parent.page_content[inizio:fine]

            overlap = conta_token(
                testo_condiviso,
                tokenizer
            )

        else:
            overlap = 0

        overlaps.append(overlap)

    print("\n--- STATISTICHE OVERLAP ---")

    if not overlaps:
        print("Nessuna coppia di child da analizzare.")
        return

    print("Coppie analizzate:", len(overlaps))
    print("Overlap minimo:", min(overlaps))
    print("Overlap massimo:", max(overlaps))
    print("Overlap medio:", statistics.mean(overlaps))
    print("Overlap mediano:", statistics.median(overlaps))
    print("Coppie senza overlap:", overlaps.count(0))


def verifica_copertura(
    parents: list[Document],
    children: list[Document]
) -> None:
    """Verifica la copertura dei caratteri non bianchi nei parent."""
    children_map = {}

    for child in children:
        parent_id = child.metadata["parent_id"]

        children_map.setdefault(parent_id, []).append(child)

    totale_caratteri = 0
    totale_coperti = 0
    parent_incompleti = 0

    for parent in parents:
        parent_id = parent.metadata["parent_id"]
        testo = parent.page_content

        maschera = [False] * len(testo)

        for child in children_map.get(parent_id, []):
            inizio = child.metadata["start_index"]
            fine = child.metadata["end_index"]

            if not 0 <= inizio < fine <= len(testo):
                raise ValueError(
                    f"Intervallo non valido: {parent_id}"
                )

            if testo[inizio:fine] != child.page_content:
                raise ValueError(
                    f"Testo non corrispondente: {parent_id}"
                )

            maschera[inizio:fine] = [True] * (fine - inizio)

        significativi = [
            i for i, carattere in enumerate(testo)
            if not carattere.isspace()
        ]

        mancanti = [
            i for i in significativi
            if not maschera[i]
        ]

        totale_caratteri += len(significativi)
        totale_coperti += len(significativi) - len(mancanti)

        if mancanti:
            parent_incompleti += 1

            print(
                f"Parent incompleto: {parent_id} | "
                f"Caratteri mancanti: {len(mancanti)}"
            )

    copertura = (
        totale_coperti / totale_caratteri * 100
        if totale_caratteri else 0
    )

    print("\n--- VERIFICA COPERTURA ---")
    print("Parent analizzati:", len(parents))
    print("Parent incompleti:", parent_incompleti)
    print("Caratteri significativi:", totale_caratteri)
    print("Caratteri coperti:", totale_coperti)
    print("Caratteri mancanti:",
          totale_caratteri - totale_coperti)
    print(f"Copertura complessiva: {copertura:.4f}%")

def verifica_integrita_tabelle(
    righe_individuate: list[dict],
    children: list[Document]
) -> None:
    """
    Verifica che le righe delle tabelle siano contenute
    integralmente nei child e individua eventuali
    frammentazioni dovute al chunking o all'overlap.
    """

    # Raggruppa i child per parent di appartenenza.
    children_map = {}

    for child in children:
        parent_id = child.metadata["parent_id"]

        children_map.setdefault(
            parent_id, []
        ).append(child)

    integre = 0
    frammentate = 0
    integre_con_frammenti = 0
    mancanti = 0

    # Set per non conteggiare più volte lo stesso child.
    child_con_frammenti = set()

    for riga in righe_individuate:

        parent_id = riga["parent_id"]
        inizio_riga = riga["inizio"]
        fine_riga = riga["fine"]

        completa = False
        parziale = False

        for child in children_map.get(parent_id, []):

            inizio_child = child.metadata["start_index"]
            fine_child = child.metadata["end_index"]

            # Verifica il contenimento completo della riga.
            if (
                inizio_child <= inizio_riga
                and fine_child >= fine_riga
            ):
                completa = True

            # Verifica una sovrapposizione soltanto parziale.
            elif (
                inizio_child < fine_riga
                and fine_child > inizio_riga
            ):
                parziale = True

                child_con_frammenti.add(
                    child.metadata["child_id"]
                )

        # Classificazione della riga.
        if completa and not parziale:
            integre += 1

        elif completa and parziale:
            integre_con_frammenti += 1

        elif parziale:
            frammentate += 1

        else:
            mancanti += 1

    totale = len(righe_individuate)

    integrita = (
        (integre + integre_con_frammenti) / totale * 100
        if totale else 0
    )

    print("\n--- INTEGRITÀ DELLE TABELLE ---")
    print("Righe analizzate:", totale)
    print("Righe completamente integre:", integre)
    print("Righe integre con frammenti:", integre_con_frammenti)
    print("Righe frammentate:", frammentate)
    print("Righe mancanti:", mancanti)

    print("Child con frammenti di righe:",
          len(child_con_frammenti))

    print(f"Integrità complessiva: {integrita:.2f}%")
