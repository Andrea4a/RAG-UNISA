"""Valutazione end-to-end: genera, revisiona, aggrega. Nessun LLM giudice."""
import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from statistics import mean

from unisa.percorsi import DOCUMENTI, DOMANDE
AZIONI = {'rispondere': 'risposta', 'astenersi': 'astensione',
          'chiedere_chiarimento': 'chiarimento'}


def carica_domande(split, excel=None):
    if split == 'prova':
        dati = json.loads((DOCUMENTI / 'ground_truth.json').read_text(encoding='utf-8'))
        if any(d['set'] != 'prova' for d in dati):
            raise ValueError('Il ground truth deve contenere solo il development set.')
        return [{'query_id': d['query_id'], 'domanda': d['domanda'],
                 'tipo_atteso': AZIONI[d['azione_attesa']],
                 'risposta_attesa': d['risposta_attesa'],
                 'fatti_attesi': [f['descrizione'] for f in d['fatti']],
                 'nota': d.get('nota_annotazione')} for d in dati]
    from openpyxl import load_workbook
    wb = load_workbook(excel or DOMANDE,
                       read_only=True, data_only=True)
    try:
        righe = iter(wb['Domande'].values)
        intestazioni = next(righe)
        out = []
        for r in righe:
            d = dict(zip(intestazioni, r))
            if str(d['Set']).strip().lower() != 'da parte':
                continue
            tipo = str(d['Tipo']).strip().lower()
            atteso = ('astensione' if tipo == 'senza risposta' else
                      'chiarimento' if tipo == 'ambigue' else 'risposta')
            out.append({'query_id': d['ID'], 'domanda': d['Domanda'],
                        'tipo_atteso': atteso, 'risposta_attesa': d['Risposta attesa'],
                        'fatti_attesi': [f.strip() for f in str(d.get('Fatti attesi (separati da |)') or '').split('|') if f.strip()],
                        'nota': d.get('Note'), 'stato_riferimento': d.get('Fatti verificati')})
        if not out:
            raise ValueError('Nessuna domanda holdout trovata nel foglio Domande.')
        return out
    finally:
        wb.close()


def percorsi(split):
    base = DOCUMENTI / f'generation_{split}'
    return (base.with_suffix('.json'), DOCUMENTI / f'generation_{split}_review.csv',
            DOCUMENTI / f'generation_{split}_summary.json')


def salva_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def fonte_record(numero, parent, descrivi):
    return {'numero': numero, 'parent_id': parent['parent_id'],
            'descrizione': descrivi(parent), 'url': parent['metadata'].get('url'),
            'testo': parent['testo']}


def esegui(args):
    from groq import Groq, APIError
    from unisa.recupero.generazione import MODELLO_GROQ, SYSTEM_PROMPT, descrivi_fonte, genera_risposta
    from unisa.recupero.pipeline import TOP_K_FINALE, RetrievalPipeline
    chiave = os.getenv('GROQ_API_KEY', '').strip()
    if not chiave:
        raise SystemExit('Imposta GROQ_API_KEY nella sessione PowerShell.')
    domande = carica_domande(args.set, args.excel)
    if len({d['query_id'] for d in domande}) != len(domande):
        raise ValueError('ID delle domande duplicati.')
    path, _, _ = percorsi(args.set)
    config = {'modello': MODELLO_GROQ, 'top_k': TOP_K_FINALE,
              'prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
              'domande_sha256': hashlib.sha256(json.dumps(domande, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'config': config, 'risultati': []}
    if data['config'] != config:
        raise SystemExit(f'Configurazione cambiata: archivia {path.name} prima di una nuova valutazione.')
    if args.retry_errors:
        data['risultati'] = [r for r in data['risultati'] if not r['errore']]
        salva_json(path, data)
    gia = {r['query_id'] for r in data['risultati']}
    restanti = [d for d in domande if d['query_id'] not in gia]
    if not restanti:
        print('Tutte le domande sono già state registrate.'); return
    pipeline = RetrievalPipeline()
    client = Groq(api_key=chiave, timeout=30.0, max_retries=0)
    for i, d in enumerate(restanti, 1):
        print(f'[{i}/{len(restanti)}] {d["query_id"]}: {d["domanda"]}', flush=True)
        record = {**d, 'tipo_generato': None, 'risposta': None, 'fonti': [], 'citazioni': [], 'errore': None}
        t = time.perf_counter()
        try:
            # Solo la domanda entra nel retrieval e nel prompt: MAI il riferimento.
            parents = pipeline.recupera_contesto(d['domanda'])
            record['latenza_retrieval'] = time.perf_counter() - t
            record['fonti'] = [fonte_record(n, p, descrivi_fonte) for n, p in enumerate(parents, 1)]
            tipo, testo, citate, stats = genera_risposta(client, d['domanda'], parents)
            record.update(tipo_generato=tipo, risposta=testo, citazioni=list(citate), **stats)
        except APIError as exc:
            record['errore'] = f'APIError: {getattr(exc, "status_code", None)}'
        except Exception as exc:
            record['errore'] = f'{type(exc).__name__}: {exc}'
        record['latenza_totale'] = time.perf_counter() - t
        data['risultati'].append(record)
        salva_json(path, data)
        print(record['errore'] or record['risposta'])
        if record['errore'] and 'APIError: 429' in record['errore']:
            # Salviamo il tentativo; non trasformiamo il rate limit in astensione.
            print('Rate limit: interruzione. Usa --retry-errors dopo il ripristino.')
            break
        if args.pausa:
            time.sleep(args.pausa)

        time.sleep(40)
    print('Risultati:', path)


def verdetto_automatico(r):
    if r['errore']:
        return 'errore_tecnico'
    if r['tipo_generato'] != r['tipo_atteso']:
        return 'errata'
    if r['tipo_atteso'] == 'astensione':
        return 'corretta'
    return ''  # Risposte fattuali e chiarimenti richiedono revisione umana.


def esporta_revisione(split):
    path, review, _ = percorsi(split)
    dati = json.loads(path.read_text(encoding='utf-8'))['risultati']
    precedenti = {}
    if review.exists():
        with review.open(encoding='utf-8-sig', newline='') as f:
            precedenti = {r['query_id']: r for r in csv.DictReader(f, delimiter=';')}
    campi = ['query_id', 'domanda', 'tipo_atteso', 'tipo_generato', 'risposta_attesa',
             'fatti_attesi', 'risposta', 'fonti_citate', 'note_riferimento', 'errore', 'verdetto', 'note_revisore']
    with review.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=campi, delimiter=';'); w.writeheader()
        for r in dati:
            old = precedenti.get(r['query_id'], {})
            auto = verdetto_automatico(r)
            # Non riutilizzare un voto se il testo generato o il riferimento è cambiato.
            stessa = all(str(old.get(k) or '') == str(r.get(k) or '') for k in ('risposta', 'risposta_attesa', 'tipo_generato'))
            w.writerow({**{k: r.get(k) for k in campi if k in r and k != 'fatti_attesi'},
                        'fatti_attesi': ' | '.join(r['fatti_attesi']),
                        'fonti_citate': ' | '.join(f"[{p['numero']}] {p['descrizione']}" for p in r['fonti'] if p['numero'] in r['citazioni']),
                        'note_riferimento': r.get('nota'),
                        'verdetto': auto or (old.get('verdetto', '') if stessa else ''),
                        'note_revisore': old.get('note_revisore', '') if stessa else ''})
    print('Apri in Excel, compila verdetto: corretta / parziale / errata (lascia vuoto se non revisionata).')
    print(review)
    print('I brani integrali sono nel JSON:', path)


def rapporto(split):
    path, review, output = percorsi(split)
    dati = json.loads(path.read_text(encoding='utf-8'))['risultati']
    with review.open(encoding='utf-8-sig', newline='') as f:
        voti = {r['query_id']: r for r in csv.DictReader(f, delimiter=';')}
    conteggi = Counter(); risposte = Counter()
    for r in dati:
        auto = verdetto_automatico(r)
        v = voti.get(r['query_id'], {})
        if not auto and v and (v.get('risposta') or '') != (r.get('risposta') or ''):
            raise ValueError('CSV di revisione non aggiornato: riesegui review.')
        voto = auto or v.get('verdetto', '').strip().lower()
        if voto not in ('', 'corretta', 'parziale', 'errata', 'errore_tecnico') or (voto == 'errore_tecnico' and not auto):
            raise ValueError(f'Verdetto non valido per {r["query_id"]}: {voto}')
        conteggi[voto or 'non_revisionata'] += 1
        if r['tipo_atteso'] == 'risposta':
            risposte[voto or 'non_revisionata'] += 1
    n = len(dati)
    ok = [r for r in dati if not r['errore']]
    def media(k):
        vals = [r[k] for r in ok if r.get(k) is not None]
        return mean(vals) if vals else None
    n_ast = sum(r['tipo_atteso'] == 'astensione' for r in dati)
    n_ch = sum(r['tipo_atteso'] == 'chiarimento' for r in dati)
    reviewed = sum(conteggi[k] for k in ('corretta', 'parziale', 'errata'))
    reviewed_answers = sum(risposte[k] for k in ('corretta', 'parziale', 'errata'))
    summary = {'set': split, 'tentativi': n, 'attese': 40 if split == 'prova' else 10,
               'conteggi': dict(conteggi), 'conteggi_domande_answerable': dict(risposte),
               'accuracy_tipo_su_tutti_i_tentativi': sum(r['tipo_generato'] == r['tipo_atteso'] and not r['errore'] for r in dati)/n if n else None,
               'accuratezza_risposte_revisionate': risposte['corretta']/reviewed_answers if reviewed_answers else None,
               'accuratezza_end_to_end_revisionate': conteggi['corretta']/reviewed if reviewed else None,
               'n_revisionate': reviewed,
               'astensioni_corrette': sum(r['tipo_atteso'] == r['tipo_generato'] == 'astensione' and not r['errore'] for r in dati),
               'astensioni_attese': n_ast,
               'chiarimenti_tipo_corretto': sum(r['tipo_atteso'] == r['tipo_generato'] == 'chiarimento' and not r['errore'] for r in dati),
               'chiarimenti_attesi': n_ch,
               'latenza_retrieval_media_s': media('latenza_retrieval'),
               'latenza_llm_media_s': media('latenza_llm'),
               'latenza_totale_media_s': media('latenza_totale'),
               'input_token_medi': media('input_tokens'), 'output_token_medi': media('output_tokens'),
               'total_token_medi': media('total_tokens')}
    salva_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if n != summary['attese'] or conteggi['non_revisionata'] or conteggi['errore_tecnico']:
        print('ATTENZIONE: valutazione non completa; le accuracy revisionate non rappresentano tutto il set.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('azione', choices=('run', 'review', 'summary'))
    parser.add_argument('--set', choices=('prova', 'holdout'), default='prova')
    parser.add_argument('--excel', type=Path, help='Percorso del file originale delle domande holdout')
    parser.add_argument('--pausa', type=float, default=0, help='Secondi tra chiamate Groq')
    parser.add_argument('--retry-errors', action='store_true', help='Rimuove i soli tentativi tecnici falliti e li riesegue')
    args = parser.parse_args()
    if args.pausa < 0: parser.error('--pausa deve essere >= 0')
    if args.azione == 'run':
        esegui(args)
    elif args.azione == 'review': esporta_revisione(args.set)
    else: rapporto(args.set)


if __name__ == '__main__':
    main()
