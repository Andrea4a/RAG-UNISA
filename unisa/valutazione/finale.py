"""Valutazione della pipeline retrieval definitiva UniSA. Solo development set.
Non modifica ChromaDB e non chiama Groq.
Uso: python -m unisa.valutazione.finale
"""
import json

from unisa.percorsi import DOCUMENTI
from unisa.valutazione import evidenze as gt
from unisa.valutazione.metriche import (
    METRICHE, SOGLIA_RILEVANZA, aggrega, crea_qrels, posizioni_gold,
    salva_csv, test_funzioni, valuta_ranking,
)

KS = (5,)

K_DENSE = 40
K_BM25 = 40
K_RRF = 40
K0 = 60
W_DENSE = 0.7
W_BM25 = 0.3
RERANKER_CONTESTUALE = True


def main():
    # Import locali: test delle metriche eseguibili anche senza i modelli.
    from unisa.indicizzazione.database import crea_client
    from unisa.indicizzazione.embedding import carica_modello, genera_embedding_query
    from unisa.recupero.dense import ricerca_dense
    from unisa.recupero.sparse import crea_indice_bm25, ricerca_bm25
    from unisa.recupero.ibrido import reciprocal_rank_fusion
    from unisa.recupero.reranking import carica_reranker, reranking

    test_funzioni()
    archivio = gt.carica_json(gt.PARENTS_PATH)
    ground_truth = gt.carica_json(gt.GROUND_TRUTH_PATH)
    gt.verifica_ground_truth(ground_truth, archivio)
    domande = [d for d in ground_truth if d['answerable'] is True]
    escluse = [d['query_id'] for d in ground_truth if d['answerable'] is not True]
    print('Domande valutate:', len(domande), '| Escluse:', escluse)
    client = crea_client()
    collection = client.get_collection(name='unisa_contestuale', embedding_function=None)
    children = gt.carica_children(collection, archivio)
    bm25, ids, documenti, metadatas = crea_indice_bm25(collection)
    modello = carica_modello()
    modello_reranker = carica_reranker()
    dettagli = []
    for i, domanda in enumerate(domande, 1):
        q = domanda['domanda']
        print(f'\n[{i}/{len(domande)}] {domanda["query_id"]}: {q}')
        qrels = crea_qrels(domanda, children)
        gold = posizioni_gold(domanda, archivio)
        if not any(g >= SOGLIA_RILEVANZA for g in qrels.values()):
            print('ATTENZIONE: nessun child sopra soglia binaria')

        def registra(nome, ranking, espandi=False):
            righe = valuta_ranking(domanda, ranking, qrels, archivio, gold, espandi)
            dettagli.extend({'query_id': domanda['query_id'], 'esperimento': nome, **r}
                            for r in righe)

        embedding = genera_embedding_query(q, modello)
        sparse = ricerca_bm25(q, bm25, ids, documenti, metadatas, k=K_BM25)
        dense = ricerca_dense(collection, embedding, k=min(K_DENSE, len(children)))
        hybrid = reciprocal_rank_fusion(dense, sparse, k0=K0, top_k=K_RRF,
                                        w_dense=W_DENSE, w_bm25=W_BM25)
        finale = reranking(q, hybrid, modello_reranker, top_k=KS[0],
                           contestuale=RERANKER_CONTESTUALE)

        # Una sola pipeline: misuriamo ciascuna fase, senza variare parametri.
        registra('Dense', dense)
        registra('BM25', sparse)
        registra('Hybrid_RRF', hybrid)
        registra('Reranker', finale)
        registra('Parent_Expansion', finale, espandi=True)

    summary = aggrega(dettagli)
    salva_csv(DOCUMENTI / 'evaluation_finale_summary.csv', summary)
    # JSON mantiene gli ID dei child e la copertura di ogni fatto.
    with (DOCUMENTI / 'evaluation_finale_details.json').open('w', encoding='utf-8') as f:
        json.dump({'config': {'collection': 'unisa_contestuale', 'ks': KS,
                   'set': 'prova', 'soglia_rilevanza': SOGLIA_RILEVANZA,
                  'k_dense': K_DENSE, 'k_bm25': K_BM25, 'k_rrf': K_RRF,
                  'k0': K0, 'w_dense': W_DENSE, 'w_bm25': W_BM25,
                  'reranker_contestuale': RERANKER_CONTESTUALE},
                  'escluse': escluse, 'summary': summary, 'dettagli': dettagli},
                  f, ensure_ascii=False, indent=2, allow_nan=False)
    print('\n--- RISULTATI MACRO @5 ---')
    print(f'{"Esperimento":32} {"Hit":>6} {"R":>6} {"P":>6} {"MRR":>6} {"nDCG":>6} '
          f'{"CtxR":>6} {"FactR":>6} {"CtxP":>6} {"IoU":>6}')
    def fmt(x):
        return f'{x:.3f}' if x is not None else 'N/A'
    for r in summary:
        if r['k'] == 5:
            print(f'{r["esperimento"]:32} ' + ' '.join(f'{fmt(r[m]):>6}' for m in METRICHE[:9]))
    print('\nRisultati salvati in:', DOCUMENTI)


if __name__ == '__main__':
    main()
