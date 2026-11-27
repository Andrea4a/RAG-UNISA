"""Suddivisione dei documenti strutturati in parent e child."""

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from unisa.indicizzazione.embedding import conta_token
from unisa.percorsi import DOCUMENTI


# Configurazione

CARTELLA_DOCUMENTI = DOCUMENTI / "strutturato"
INVENTARIO_PATH = DOCUMENTI / "inventario.json"

# Livelli della gerarchia Markdown utilizzati per lo splitting.
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# Pattern per individuare i marcatori delle pagine PDF.
# Il gruppo di cattura restituisce il numero della pagina.
PATTERN_PAGINA = re.compile(
    r"<!--[ \t]*Pagina PDF[ \t]+(\d+)[ \t]*-->"
)

# Separatore utilizzato per ricomporre i segmenti testuali.
SEPARATORE = "\n\n"

# Dimensioni espresse in token.
CHUNK_SIZE = 128
CHUNK_OVERLAP = 20

# Separatori in ordine di priorità.
SEPARATORS = ["\n\n", "\n", " ", ""]


# Lettura degli input


def carica_inventario(percorso: Path) -> dict:
    """Indicizza i metadati dell’inventario per ID del documento."""
    dati = json.loads(percorso.read_text(encoding="utf-8"))

    return {doc["id"]: doc for doc in dati}


def carica_righe_tabelle(percorso: Path) -> list[dict]:
    """Legge le sole righe di tabella dai paragrafi in formato JSONL."""
    righe_tabelle = []

    with percorso.open("r", encoding="utf-8") as file:
        for riga in file:
            if not riga.strip():
                continue

            paragrafo = json.loads(riga)

            if paragrafo["tipo"] == "tabella":
                righe_tabelle.append(paragrafo)

    return righe_tabelle


def rimuovi_frontmatter(testo: str) -> str:
    """Rimuove il frontmatter YAML iniziale dal Markdown."""
    righe = testo.splitlines(keepends=True)

    if not righe or righe[0].strip() != "---":
        return testo.strip()

    for i, riga in enumerate(righe[1:], start=1):
        if riga.strip() == "---":
            return "".join(righe[i + 1:]).strip()

    raise ValueError("Frontmatter non chiuso correttamente")


# Riferimenti alle pagine PDF


def elabora_pagine(
    testo: str,
    pagina_corrente: int | None
) -> tuple[str, list[dict], int | None]:
    """Rimuove i marcatori PDF e restituisce testo, mappa dei caratteri e ultima pagina."""
    segmenti = []
    mappa_pagine = []

    posizione = 0

    parti = PATTERN_PAGINA.split(testo)

    for indice, parte in enumerate(parti):
        # Gli indici dispari contengono i numeri delle pagine.
        if indice % 2 == 1:
            pagina_corrente = int(parte)
            continue

        segmento = parte.strip()

        if not segmento:
            continue

        if segmenti:
            posizione += len(SEPARATORE)

        inizio = posizione
        posizione += len(segmento)

        segmenti.append(segmento)

        mappa_pagine.append({
            "inizio": inizio,
            "fine": posizione,
            "pagina": pagina_corrente
        })

    testo_pulito = SEPARATORE.join(segmenti)

    return testo_pulito, mappa_pagine, pagina_corrente


def trova_pagine(
    inizio_child: int,
    fine_child: int,
    mappa_pagine: list[dict]
) -> list[int]:
    """Restituisce le pagine PDF che si sovrappongono a un intervallo di caratteri."""
    pagine = set()

    for segmento in mappa_pagine:
        inizio_pagina = segmento["inizio"]
        fine_pagina = segmento["fine"]
        pagina = segmento["pagina"]

        if inizio_child < fine_pagina and fine_child > inizio_pagina:
            if pagina is not None:
                pagine.add(pagina)

    return sorted(pagine)


# Creazione dei parent


def crea_parents(
    cartella: Path,
    inventario: dict
) -> list[Document]:
    """Suddivide il Markdown per titoli e associa metadati e pagine a ogni parent."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=True
    )

    all_parents = []

    files = sorted(cartella.glob("*.md"))

    if not files:
        raise FileNotFoundError(
            f"Nessun documento Markdown trovato in {cartella}"
        )

    for file in files:
        doc_id = file.stem

        if doc_id not in inventario:
            raise KeyError(
                f"Documento {doc_id} assente dall'inventario"
            )

        testo = file.read_text(encoding="utf-8")

        contenuto = rimuovi_frontmatter(testo)

        metadati = inventario[doc_id]

        parents = splitter.split_text(contenuto)

        # La pagina corrente viene mantenuta tra parent
        # appartenenti allo stesso documento.
        pagina_corrente = None

        parents_puliti = []

        for i, parent in enumerate(parents, start=1):
            pulito, mappa, pagina_corrente = elabora_pagine(
                parent.page_content,
                pagina_corrente
            )

            # Il parent viene scartato soltanto dopo aver
            # aggiornato l'eventuale riferimento alla pagina.
            if not pulito:
                continue

            parent.page_content = pulito

            parent.metadata.update(metadati)

            parent.metadata["parent_id"] = (
                f"{doc_id}_P{i:03d}"
            )

            parent.metadata["mappa_pagine"] = mappa

            parents_puliti.append(parent)

        all_parents.extend(parents_puliti)

        rimossi = len(parents) - len(parents_puliti)

        print(
            f"{file.name:8} | "
            f"Originali: {len(parents):3} | "
            f"Validi: {len(parents_puliti):3} | "
            f"Rimossi: {rimossi:3}"
        )

    return all_parents


def verifica_parents(parents: list[Document]) -> None:
    """Controlla contenuto, identificativi e riferimenti alle pagine dei parent."""
    if not parents:
        raise ValueError("Nessun parent valido generato")

    ids = set()
    pagine_mancanti = set()

    for parent in parents:
        testo = parent.page_content
        metadati = parent.metadata

        parent_id = metadati["parent_id"]
        mappa = metadati["mappa_pagine"]

        if not testo.strip():
            raise ValueError(f"Parent vuoto: {parent_id}")

        if parent_id in ids:
            raise ValueError(f"ID duplicato: {parent_id}")

        ids.add(parent_id)

        if PATTERN_PAGINA.search(testo):
            raise ValueError(
                f"Marcatore residuo: {parent_id}"
            )

        is_pdf = Path(metadati["file"]).suffix.lower() == ".pdf"

        if not mappa:
            raise ValueError(f"Mappa vuota: {parent_id}")

        for segmento in mappa:
            inizio = segmento["inizio"]
            fine = segmento["fine"]

            if not 0 <= inizio < fine <= len(testo):
                raise ValueError(
                    f"Intervallo non valido: {parent_id}"
                )

            if is_pdf and segmento["pagina"] is None:
                pagine_mancanti.add(parent_id)

    print("\n--- VERIFICA INTEGRITÀ ---")

    print("Identificativi univoci: OK")
    print("Assenza di parent vuoti: OK")
    print("Assenza di marcatori residui: OK")
    print("Validità degli intervalli: OK")

    if pagine_mancanti:
        print("ATTENZIONE - Riferimenti alle pagine mancanti:")

        for parent_id in sorted(pagine_mancanti):
            print(f"  - {parent_id}")

    else:
        print("Riferimenti alle pagine PDF: completi")


# Creazione dei child


def crea_child_splitter(tokenizer) -> RecursiveCharacterTextSplitter:
    """Configura lo splitter ricorsivo con lunghezze in token."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=lambda testo: conta_token(testo, tokenizer)
    )


def crea_children(
    parents: list[Document],
    tokenizer
) -> list[Document]:
    """Suddivide i parent e registra per ogni child posizione, pagine e token."""
    splitter = crea_child_splitter(tokenizer)

    all_children = []

    for parent in parents:
        parent_id = parent.metadata["parent_id"]
        mappa_pagine = parent.metadata["mappa_pagine"]

        children = splitter.split_documents([parent])
        testo_parent = parent.page_content
        inizio_precedente = -1
        fine_precedente = 0

        for i, child in enumerate(children, start=1):
            child.metadata["child_id"] = (
                f"{parent_id}_C{i:03d}"
            )

            # Gli indici sono in caratteri, mentre l'overlap dello splitter è in token.
            # Il child può sovrapporsi al precedente, ma deve estenderne la fine.
            lunghezza = len(child.page_content)
            minimo = max(inizio_precedente + 1, fine_precedente - lunghezza + 1)
            inizio = testo_parent.find(child.page_content, minimo)
            if inizio == -1:
                raise ValueError(
                    f"Posizione non valida: {child.metadata['child_id']}"
                )
            fine = inizio + lunghezza
            if inizio > fine_precedente:
                testo_saltato = testo_parent[fine_precedente:inizio]

                if testo_saltato.strip():
                    raise ValueError(
                        f"Contenuto saltato: {child.metadata['child_id']}"
                    )
            inizio_precedente, fine_precedente = inizio, fine

            pagine = trova_pagine(
                inizio,
                fine,
                mappa_pagine
            )

            child.metadata["start_index"] = inizio
            child.metadata["end_index"] = fine
            child.metadata["pagine_child"] = pagine

            child.metadata["numero_token"] = conta_token(
                child.page_content,
                tokenizer
            )

            all_children.append(child)

    return all_children


# Posizioni delle righe di tabella


def individua_righe_tabelle(
    parents: list[Document],
    righe_tabelle: list[dict]
) -> list[dict]:
    """Associa ogni riga di tabella alla sua posizione nel parent, se univoca."""
    righe_individuate = []

    for riga in righe_tabelle:
        corrispondenze = []

        for parent in parents:
            meta = parent.metadata

            if meta["id"] != riga["doc"]:
                continue

            sezione = [
                meta[h]
                for h in ["H1", "H2", "H3"]
                if h in meta
            ]

            if sezione != riga["sezione"]:
                continue

            posizione = 0

            for linea in parent.page_content.splitlines(
                keepends=True
            ):
                if linea.strip() == riga["testo"]:
                    inizio = posizione + (
                        len(linea) - len(linea.lstrip())
                    )

                    fine = inizio + len(riga["testo"])

                    pagine = trova_pagine(
                        inizio,
                        fine,
                        meta["mappa_pagine"]
                    )

                    if (
                        riga["pagina"] is None
                        or riga["pagina"] in pagine
                    ):
                        corrispondenze.append({
                            "parent_id": meta["parent_id"],
                            "inizio": inizio,
                            "fine": fine,
                            "testo": riga["testo"]
                        })

                posizione += len(linea)

        if len(corrispondenze) == 1:
            righe_individuate.append(corrispondenze[0])

        else:
            print(
                "Riga non individuata univocamente:",
                riga["doc"],
                riga["testo"],
                "Corrispondenze:", len(corrispondenze)
            )

    print("\n--- INDIVIDUAZIONE TABELLE ---")
    print("Righe attese:", len(righe_tabelle))
    print("Righe individuate:", len(righe_individuate))
    print("Righe non individuate univocamente:",
          len(righe_tabelle) - len(righe_individuate))

    return righe_individuate
