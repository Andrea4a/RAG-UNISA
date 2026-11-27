"""Metriche condivise per ranking e copertura delle evidenze."""

import csv
import math
from statistics import mean

from unisa.valutazione import evidenze as gt

KS = (1, 3, 5, 10, 20)
SOGLIA_RILEVANZA = 0.5


def evidenze_domanda(domanda):
    return [e for f in domanda['fatti'] for e in f['evidenze_alternative']]


def crea_qrels(domanda, children):
    """Grado di rilevanza = massima frazione di una singola evidenza coperta.
    Le alternative equivalenti restano documenti distinti per le metriche ranking.
    """
    per_parent = {}
    for e in evidenze_domanda(domanda):
        per_parent.setdefault(e['parent_id'], []).append(e)
    qrels = {}
    for child in children:
        pid = child['metadata']['parent_id']
        grado = max((gt.calcola_copertura(child, e, ignora_spazi=True)
                     for e in per_parent.get(pid, [])), default=0.0)
        qrels[child['child_id']] = grado
    return qrels


def dcg(gradi):
    # Guadagno lineare: i gradi sono frazioni continue in [0, 1].
    return sum(g / math.log2(i + 2) for i, g in enumerate(gradi))


def metriche_ranking(risultati, qrels, k, soglia=SOGLIA_RILEVANZA):
    if k <= 0 or not 0 < soglia <= 1:
        raise ValueError('k o soglia non validi')
    ids = [r['child_id'] for r in risultati[:k]]
    if len(ids) != len(set(ids)):
        raise ValueError('Ranking con child duplicati')
    gradi = [qrels[cid] for cid in ids]
    rilevanti = {cid for cid, g in qrels.items() if g >= soglia}
    binari = [g >= soglia for g in gradi]
    hit = sum(binari)
    primo = next((i for i, v in enumerate(binari, 1) if v), None)
    ideale = dcg(sorted(qrels.values(), reverse=True)[:k])
    # Le query prive di child rilevanti secondo la soglia sono escluse
    # dalle macro-medie binarie (None), non dal contesto o dal NDCG.
    return {
        'hit_rate': float(hit > 0) if rilevanti else None,
        'recall': hit / len(rilevanti) if rilevanti else None,
        'precision': hit / k if rilevanti else None,
        'mrr': (1 / primo if primo else 0.0) if rilevanti else None,
        'ndcg': dcg(gradi) / ideale if ideale else None,
        'n_rilevanti': len(rilevanti),
        'n_restituiti': len(ids),
    }


def posizioni_significative(intervalli, archivio):
    """Posizioni (parent_id, offset) distinte, escludendo whitespace."""
    posizioni = set()
    for pid, righe in intervalli.items():
        testo = archivio[pid]['testo']
        for a, b in gt.unisci_intervalli(righe):
            if not 0 <= a < b <= len(testo):
                raise ValueError(f'Intervallo fuori dal parent: {pid} {a}:{b}')
            posizioni.update((pid, i) for i in range(a, b)
                             if not testo[i].isspace())
    return posizioni


def posizioni_gold(domanda, archivio):
    intervalli = {}
    for e in evidenze_domanda(domanda):
        intervalli.setdefault(e['parent_id'], []).append(
            (e['start_index'], e['end_index']))
    return posizioni_significative(intervalli, archivio)


def metriche_contesto(domanda, intervalli, archivio, gold=None):
    """Recall macro sui fatti (OR tra alternative); precision e IoU posizionali.
    IoU confronta con l'unione di TUTTE le evidenze annotate, non con un
    sottoinsieme ottimo: puo penalizzare le alternative non recuperate.
    """
    copertura = gt.valuta_copertura_domanda(domanda, intervalli, ignora_spazi=True)
    if copertura is None:
        return None
    if gold is None:
        gold = posizioni_gold(domanda, archivio)
    recuperate = posizioni_significative(intervalli, archivio)
    intersezione = len(recuperate & gold)
    unione = len(recuperate | gold)
    return {
        **copertura,
        'context_precision': intersezione / len(recuperate) if recuperate else 0.0,
        'iou': intersezione / unione if unione else 0.0,
        'context_recall_posizionale': intersezione / len(gold) if gold else None,
        'caratteri_contesto': len(recuperate),
        'caratteri_rilevanti': intersezione,
    }


def valuta_ranking(domanda, risultati, qrels, archivio, gold, espandi=False, *, ks=KS):
    """Per ogni k valuta il ranking child e il contesto selezionato."""
    righe = []
    for k in ks:
        top = risultati[:k]
        if espandi:
            # Espansione equivalente a espandi_parents, limitata agli intervalli.
            pids = dict.fromkeys(r['metadata']['parent_id'] for r in top)
            intervalli = {pid: [(0, len(archivio[pid]['testo']))] for pid in pids}
        else:
            intervalli = gt.raggruppa_intervalli(top)
        ranking = metriche_ranking(top, qrels, k)
        contesto = metriche_contesto(domanda, intervalli, archivio, gold)
        righe.append({'k': k, **ranking, **contesto,
                      'n_parents': len(intervalli),
                      'child_ids': [r['child_id'] for r in top]})
    return righe


METRICHE = ('hit_rate', 'recall', 'precision', 'mrr', 'ndcg',
            'context_recall', 'strict_fact_recall', 'context_precision',
            'iou', 'context_recall_posizionale', 'caratteri_contesto',
            'caratteri_rilevanti', 'n_restituiti', 'n_parents')


def aggrega(dettagli):
    gruppi = {}
    for r in dettagli:
        gruppi.setdefault((r['esperimento'], r['k']), []).append(r)
    summary = []
    for (nome, k), righe in gruppi.items():
        record = {'esperimento': nome, 'k': k, 'n_domande': len(righe),
                  'n_domande_ranking_binario': sum(r['n_rilevanti'] > 0 for r in righe)}
        for m in METRICHE:
            valori = [r[m] for r in righe if r[m] is not None]
            record[m] = mean(valori) if valori else None
        summary.append(record)
    return summary


def salva_csv(percorso, righe):
    with percorso.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(righe[0]))
        writer.writeheader()
        writer.writerows(righe)


def test_funzioni():
    archivio = {'P': {'testo': 'abcdefghij'}, 'Q': {'testo': 'abcdefghij'}}
    e1 = {'parent_id': 'P', 'start_index': 0, 'end_index': 4, 'testo': 'abcd'}
    e2 = {**e1, 'parent_id': 'Q'}
    domanda = {'query_id': 'test', 'answerable': True, 'fatti': [
        {'fact_id': 'F01', 'evidenze_alternative': [e1, e2]}]}
    def child(cid, pid, a, b):
        return {'child_id': cid, 'metadata': {'parent_id': pid,
                'start_index': a, 'end_index': b}, 'testo': archivio[pid]['testo'][a:b]}
    children = [child('a', 'P', 0, 2), child('b', 'P', 2, 6), child('c', 'Q', 5, 10)]
    qrels = crea_qrels(domanda, children)
    assert qrels == {'a': 0.5, 'b': 0.5, 'c': 0.0}
    r = metriche_ranking([children[2], children[0]], qrels, 2)
    assert r['recall'] == .5 and r['precision'] == .5 and r['mrr'] == .5
    assert 0 < r['ndcg'] < 1
    r = metriche_ranking([children[0]], qrels, 5)
    assert r['precision'] == .2
    c = metriche_contesto(domanda, gt.raggruppa_intervalli(children[:2]), archivio)
    assert c['context_recall'] == 1 and c['strict_fact_recall'] == 1
    assert c['context_precision'] == 4/6 and c['iou'] == 4/10
    c = metriche_contesto(domanda, {'P': [(0, 2)], 'Q': [(2, 4)]}, archivio)
    assert c['context_recall'] == .5 and c['strict_fact_recall'] == 0
    assert metriche_ranking([], {'a': .1}, 5)['recall'] is None
    assert metriche_ranking([], {'a': .1}, 5)['ndcg'] == 0
    print('Test metriche: OK')
