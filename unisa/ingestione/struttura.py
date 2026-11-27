"""
Fermata 7 della linea: rendere esplicita la struttura.

Legge  documenti/ricomposto/Dxx.md   (uscita della fermata 6)
Scrive documenti/strutturato/Dxx.md  (titoli in Markdown: #, ##, ###)
       documenti/paragrafi.jsonl     (un paragrafo per riga, con i suoi metadati: documento, sezione, pagina, tipo)

PDF (il Regolamento):
  - i titoli numerati diventano titoli Markdown, con il livello preso dalla numerazione:
    «3. QUANTO PAGARE» → #, «3.3 STUDENTI…» → ##, «2.2.1 Corsi di Laurea…» → ###        (quaderno B3, B7 passo 4)
  - il titolo del documento ripetuto in cima («REGOLAMENTO IN MATERIA…», «Anno Accademico…») si toglie:
    è già nei metadati
  - il segnaposto di pagina diventa il metadato «pagina» di ogni paragrafo
Pagine web:
  - il «#» con il percorso del sito («Servizi ed Informazioni ❭ Tasse e Contributi») diventa il titolo vero («# Tasse e Contributi»)
  - i «##» generici, uguali in più pagine («I servizi della segreteria studenti») o che ripetono il titolo, si tolgono

Ogni paragrafo riceve la sua sezione: il percorso dei titoli sopra di lui («1. DISPOSIZIONI GENERALI › 1.3 ATTESTAZIONE ISEE…»).
È quello che useranno il chunking (fermata 8), il prefisso «documento · sezione» (fermata 9) e le citazioni.

Uso:  python -m unisa.ingestione.struttura
"""
import json, re
from collections import Counter

import pandas as pd

from unisa.ingestione.pulizia import norm, separa_intestazione, LETTERE
from unisa.ingestione.ricomposizione import tipo, TITOLO, SEGNAPOSTO

from unisa.percorsi import DOCUMENTI, DOMANDE, INVENTARIO_EXCEL

CARTELLA = DOCUMENTI
RICOMPOSTO, STRUTTURATO = CARTELLA / "ricomposto", CARTELLA / "strutturato"


def metadati_file(testa):
    return dict(re.findall(r"^(\w+): (.*)$", testa, re.M))


def livello_numerato(titolo):
    """«3.» → 1, «3.3» → 2, «2.2.1» → 3."""
    numero = titolo.split()[0].rstrip(".")
    return min(numero.count(".") + 1, 4)


def struttura_pdf(corpo, conta):
    blocchi = [b.strip() for b in re.split(r"\n\s*\n", corpo) if b.strip()]
    out, pagina, visto_numerato = [], None, False
    for b in blocchi:
        m = SEGNAPOSTO.match(b)
        if m:
            pagina = int(m.group(1)); out.append(("pagina", pagina, b)); continue
        t = tipo(b)
        if t == "titolo" and TITOLO.match(b):
            visto_numerato = True
            out.append(("titolo", livello_numerato(b), b)); conta["titoli numerati"] += 1
        elif t == "titolo" and not visto_numerato:
            conta["titolo del documento tolto (è nei metadati)"] += 1      # «REGOLAMENTO IN MATERIA…» in cima
        elif not visto_numerato and b.startswith("Anno Accademico"):
            conta["titolo del documento tolto (è nei metadati)"] += 1
        else:
            out.append(("testo" if t == "titolo" else t, pagina, b))   # una riga maiuscola senza numero in mezzo al testo resta testo
    return out


def struttura_web(corpo, conta, generici):
    out = []
    for b in [b.strip() for b in re.split(r"\n\s*\n", corpo) if b.strip()]:
        m = re.match(r"^(#+)\s+(.*)$", b)
        if not m:
            out.append(("voce" if b.startswith("- ") else "tabella" if " | " in b else "testo", None, b)); continue
        livello, testo = len(m.group(1)), m.group(2).strip()
        if livello == 1 and "❭" in testo:
            testo = testo.split("❭")[-1].strip(); conta["titolo della pagina ripulito dal percorso"] += 1
        titolo_pagina = next((x[2] for x in out if x[0] == "titolo" and x[1] == 1), "")
        if livello == 2 and (testo in generici or (titolo_pagina and titolo_pagina.lower() in testo.lower())):
            conta["titoli generici tolti"] += 1; continue
        out.append(("titolo", livello, testo))
    # livelli compatti: se una pagina passa da # a ### senza ##, il ### diventa ## (e così via),
    # così lo splitter per titoli della fermata 8 trova le sezioni a tutti i livelli
    usati = sorted({x for t, x, _ in out if t == "titolo"})
    nuovo = {v: i + 1 for i, v in enumerate(usati)}
    if any(k != v for k, v in nuovo.items()):
        conta["livelli dei titoli compattati"] += 1
    return [(t, nuovo[x] if t == "titolo" else x, b) for t, x, b in out]


def scrivi(did, meta, elementi, f_md, f_jsonl, conta):
    righe, percorso, pagina = [], {}, None
    for t, x, testo in elementi:
        if t == "pagina":
            pagina = x; righe.append(testo); continue
        if t == "titolo":
            percorso = {k: v for k, v in percorso.items() if k < x}; percorso[x] = testo
            righe.append("#" * x + " " + testo)
            continue
        if t in ("testo", "voce", "tabella"):
            pagina = x if x is not None else pagina
        sezione = [percorso[k] for k in sorted(percorso)]
        if not sezione: conta["paragrafi senza sezione"] += 1
        f_jsonl.write(json.dumps({"doc": did, "documento": meta.get("documento", ""), "url": meta.get("url", ""),
                                  "versione": meta.get("versione", ""), "sezione": sezione, "pagina": pagina,
                                  "tipo": t, "testo": testo}, ensure_ascii=False) + "\n")
        conta["paragrafi"] += 1
        righe.append(testo)
    f_md.write_text("\n\n".join(righe) + "\n", encoding="utf-8")


def main():
    STRUTTURATO.mkdir(parents=True, exist_ok=True)
    file = sorted(RICOMPOSTO.glob("D*.md"))
    # «##» generici: uguali in almeno due pagine web
    h2 = Counter()
    for f in file:
        corpo = separa_intestazione(f.read_text(encoding="utf-8"))[1]
        if not SEGNAPOSTO.search(corpo):
            h2.update({m.strip() for m in re.findall(r"^## (.*)$", corpo, re.M)})
    generici = {t for t, n in h2.items() if n >= 2}
    testi = {}
    with open(CARTELLA / "paragrafi.jsonl", "w", encoding="utf-8") as fj:
        for f in file:
            testa, corpo = separa_intestazione(f.read_text(encoding="utf-8"))
            meta, conta = metadati_file(testa), Counter()
            pdf = any(SEGNAPOSTO.match(r.strip()) for r in corpo.split("\n"))
            elementi = struttura_pdf(corpo, conta) if pdf else struttura_web(corpo, conta, generici)
            dest = STRUTTURATO / f.name
            scrivi(f.stem, meta, elementi, dest, fj, conta)
            dest.write_text(testa + "\n" + dest.read_text(encoding="utf-8"), encoding="utf-8")
            testi[f.stem] = norm(dest.read_text(encoding="utf-8"))
            livelli = Counter(x for t, x, _ in elementi if t == "titolo")
            print(f"{f.stem}: {conta['paragrafi']} paragrafi · titoli per livello {dict(sorted(livelli.items()))}"
                  + "".join(f" · {k}: {v}" for k, v in conta.items() if k != "paragrafi" and v))
    if generici: print("Titoli «##» generici tolti dalle pagine web:", sorted(generici))

    # l'albero dei titoli del Regolamento, per il controllo a occhio
    d01 = STRUTTURATO / "D01.md"
    if d01.exists():
        print("\nAlbero dei titoli di D01 (i primi 25):")
        for r in [r for r in d01.read_text(encoding="utf-8").split("\n") if r.startswith("#")][:25]:
            print("   " + "  " * (len(r) - len(r.lstrip("#")) - 1) + r.lstrip("#").strip()[:80])

    # i fatti ci sono ancora?
    inv = pd.read_excel(INVENTARIO_EXCEL, sheet_name="Inventario", dtype=str).fillna("")
    url_a_id = {norm(u): i for i, u in zip(inv["ID"], inv["URL"])}
    dom = pd.read_excel(DOMANDE, sheet_name="Domande", dtype=str).fillna("")
    mancano, controllate = [], 0
    for _, q in dom.iterrows():
        if not q["Fatti verificati"].startswith("sì") or not q["Fatti attesi (separati da |)"]: continue
        fonti = [url_a_id.get(norm(q["URL della fonte"]))] if q["URL della fonte"] else [LETTERE[x] for x in re.findall(r"\b[RTPCF]\b", q["Fonte"])]
        fonti = [x for x in fonti if x in testi]
        if not fonti: continue
        controllate += 1
        for fa in q["Fatti attesi (separati da |)"].split("|"):
            if fa.strip() and not any(norm(fa) in testi[x] for x in fonti): mancano.append((q["ID"], fa.strip()))
    print(f"\n== Dopo la struttura ==\nDomande controllate: {controllate}. Fatti persi: {len(mancano)}.")
    for qid, fa in mancano: print(f"  {qid} «{fa}»")


if __name__ == "__main__":
    main()
