"""Recupero ibrido, reranking ed espansione del contesto."""

from unisa.indicizzazione.archivio import carica_parents, espandi_parents
from unisa.indicizzazione.database import crea_client
from unisa.indicizzazione.embedding import carica_modello
from unisa.recupero import TOP_K_FINALE
from unisa.recupero.ibrido import ricerca_hybrid
from unisa.recupero.reranking import carica_reranker, reranking
from unisa.recupero.sparse import crea_indice_bm25


class RetrievalPipeline:

    def __init__(self):
        self.client = crea_client()

        self.collection = self.client.get_collection(
            name="unisa_contestuale",
            embedding_function=None
        )

        self.modello = carica_modello()
        self.modello_reranker = carica_reranker()

        (
            self.bm25,
            self.ids,
            self.documenti,
            self.metadatas
        ) = crea_indice_bm25(self.collection)

        self.archivio = carica_parents()

    def recupera_contesto(self, domanda, top_k=TOP_K_FINALE):
        risultati_hybrid = ricerca_hybrid(
            domanda,
            self.modello,
            self.collection,
            self.bm25,
            self.ids,
            self.documenti,
            self.metadatas,
            k_dense=40,
            k_bm25=40,
            k0=60,
            top_k=40
        )

        risultati_finali = reranking(
            domanda,
            risultati_hybrid,
            self.modello_reranker,
            top_k=top_k,
            contestuale=True
        )

        return espandi_parents(risultati_finali, self.archivio)
