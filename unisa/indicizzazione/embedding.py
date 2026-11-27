import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

MODEL_NAME = "intfloat/multilingual-e5-small"

def crea_prefisso_contestuale(child: Document) -> str:

    metadata = child.metadata
    parti = []

    documento = metadata.get("documento")
    versione = metadata.get("versione")

    if documento:
        parti.append(f"Documento: {documento}")

    if versione:
        parti.append(f"Versione: {versione}")

    gerarchia = [
        metadata[h]
        for h in ["H1", "H2", "H3"]
        if metadata.get(h)
    ]

    if gerarchia:
        sezione = " > ".join(gerarchia)
        parti.append(f"Sezione: {sezione}")

    return "\n".join(parti)

def crea_testo_embedding(
    child: Document,
    contestuale: bool = False
) -> str:

    testo = child.page_content

    if contestuale:
        prefisso = crea_prefisso_contestuale(child)

        if prefisso:
            testo = f"{prefisso}\n\n{testo}"

    return f"passage: {testo}"

def carica_modello() -> SentenceTransformer:
    """Carica il modello di embedding preaddestrato."""

    model = SentenceTransformer(MODEL_NAME)

    return model


def genera_embedding(
    children: list[Document],
    modello,
    contestuale: bool = False,
    batch_size: int = 32
) -> np.ndarray:

    testi = [
        crea_testo_embedding(
            child,
            contestuale=contestuale
        )
        for child in children
    ]

    embeddings = modello.encode(
        testi,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings

def conta_token(testo: str, tokenizer) -> int:
    """
    Restituisce il numero di token presenti nel testo,
    senza aggiungere token speciali e senza troncamento.
    """
    return len(tokenizer.encode(
        testo,
        add_special_tokens=False,
        truncation=False
    ))

def verifica_input_embedding(
    children: list[Document],
    tokenizer,
    contestuale: bool = False,
    limite: int = 512
) -> None:

    lunghezze = []
    fuori_limite = []

    for child in children:

        testo = crea_testo_embedding(
            child,
            contestuale=contestuale
        )

        token = tokenizer.encode(
            testo,
            add_special_tokens=True,
            truncation=False
        )

        lunghezza = len(token)
        lunghezze.append(lunghezza)

        if lunghezza > limite:
            fuori_limite.append(
                child.metadata["child_id"]
            )

    print("\n--- VERIFICA INPUT EMBEDDING ---")
    print("Prefisso contestuale:", contestuale)
    print("Input analizzati:", len(lunghezze))
    print("Token minimi:", min(lunghezze))
    print("Token massimi:", max(lunghezze))
    print("Token medi:", sum(lunghezze) / len(lunghezze))
    print("Input fuori limite:", len(fuori_limite))

    if fuori_limite:
        raise ValueError(
            f"Input troppo lunghi: {fuori_limite}"
        )

def verifica_embedding(
    embeddings: np.ndarray,
    numero_child: int,
    dimensione: int = 384
) -> None:

    if embeddings.shape != (numero_child, dimensione):
        raise ValueError("Shape degli embedding non valida")

    if not np.isfinite(embeddings).all():
        raise ValueError("Presenti valori NaN o infiniti")

    norme = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norme, 1.0, atol=1e-6):
        raise ValueError("Embedding non normalizzati")

    print("\n--- VERIFICA EMBEDDING ---")
    print("Shape:", embeddings.shape)
    print("Norma minima:", norme.min())
    print("Norma massima:", norme.max())
    print("Norma media:", norme.mean())
    print("Embedding validi:", numero_child)

def genera_embedding_query(domanda, modello):

    testo = "query: " + domanda

    embedding = modello.encode(
        testo,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embedding
    
if __name__ == "__main__":

    model = carica_modello()

    # Recupera il tokenizer associato al modello.
    tokenizer = model.tokenizer

    print("Modello:", MODEL_NAME)
    print("Tokenizer:", type(tokenizer).__name__)
    print("Dimensione vocabolario:", tokenizer.vocab_size)
    print("Massimo numero di token:", model.max_seq_length)
    print("Dimensione embedding:",
          model.get_embedding_dimension())

        # Frase di esempio
    testo = "Lo studente deve presentare l'attestazione ISEE."

    # Suddivisione del testo in token
    tokens = tokenizer.tokenize(testo)

    # Conversione dei token nei rispettivi identificativi
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("\nTesto:", testo)
    print("Token:", tokens)
    print("Token ID:", token_ids)
    print("Numero di token:", len(tokens))
    print("Numero di caratteri:", len(testo))

