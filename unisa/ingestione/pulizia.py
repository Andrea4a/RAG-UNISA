"""
Fermata 5 della linea: pulire il rumore, senza togliere la struttura.

Legge  documenti/testo/Dxx.md   (uscita della fermata 4)
Scrive documenti/pulito/Dxx.md  e  documenti/pulizia.json (cosa è stato tolto, documento per documento)

1. Pulizia generale, uguale per tutti i documenti                     (quaderno B7 passo 2, linea fermata 5)
   - caratteri: spazi non separabili, trattini morbidi, legature, spazi doppi
   - PDF: righe ripetute su molte pagine («Pagina 4 di 19»), parole sillabate a fine riga, sommario con i puntini
   - pagine web: briciole di pane e «Condividi» prima del titolo, menu degli anni accademici,
     titoli vuoti, righe-link («… | PDF Altri formati»), «Indietro», questionario, celle ripetute
2. Regole decise da te, documento per documento, nel file regole_pulizia.json
   (frasi superate, contrasti tra fonti, copertina, formula spezzata, nota a piè di pagina)
3. Controllo: ogni regola deve aver trovato il suo testo, e i fatti delle domande di prova devono esserci ancora.

Uso:  python -m unisa.ingestione.pulizia
"""
import json, re, unicodedata
from collections import Counter

import pandas as pd

from unisa.percorsi import DOCUMENTI, DOMANDE, INVENTARIO_EXCEL, REGOLE_PULIZIA

CARTELLA = DOCUMENTI
TESTO, PULITO = CARTELLA / "testo", CARTELLA / "pulito"
SEGNAPOSTO = re.compile(r"^<!-- Pagina PDF \d+ -->$")
LETTERE = {"R": "D01", "T": "D02", "P": "D03", "C": "D04", "F": "D05"}


def norm(t):
    t = unicodedata.normalize("NFC", t).lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t).strip()


def separa_intestazione(testo):
    """Il blocco --- … --- in cima (id, url, versione) non si pulisce."""
    m = re.match(r"(---\n.*?\n---\n)", testo, re.S)
    return (m.group(1), testo[m.end():]) if m else ("", testo)


# ---------- 1. pulizia generale ----------
def caratteri(t, conta):
    prima = t
    t = unicodedata.normalize("NFC", t).replace("\u00a0", " ").replace("\u00ad", "")
    for a, b in {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}.items():
        t = t.replace(a, b)
    t = "\n".join(re.sub(r"[ \t]+", " ", r).rstrip() for r in t.split("\n"))
    conta["spazi e caratteri speciali tolti"] += len(prima) - len(t)
    return t


def righe_ripetute_pdf(righe, conta, soglia=0.5):
    """Righe che compaiono (a meno dei numeri) su almeno metà delle pagine: intestazioni e piè di pagina."""
    pagine = [set() for _ in range(sum(bool(SEGNAPOSTO.match(r)) for r in righe))]
    k = -1
    for r in righe:
        if SEGNAPOSTO.match(r): k += 1
        elif k >= 0 and r.strip(): pagine[k].add(re.sub(r"\d+", "#", r.strip()))
    if len(pagine) < 3:
        return righe
    freq = Counter(x for p in pagine for x in p)
    ripetute = {x for x, n in freq.items() if n >= soglia * len(pagine)}
    tenute = [r for r in righe if SEGNAPOSTO.match(r) or re.sub(r"\d+", "#", r.strip()) not in ripetute]
    conta["righe ripetute su ogni pagina"] += len(righe) - len(tenute)
    conta["esempi ripetute"] = sorted(ripetute)[:5]
    return tenute


def sillabe(t, conta):
    """«contri-\\nbuzione» → «contribuzione»: trattino in fondo alla riga e minuscola all'inizio della successiva."""
    trovate = re.findall(r"(\w+)-\n([a-zà-ù]\w*)", t)
    conta["parole sillabate ricucite"] += len(trovate)
    if trovate:
        conta["esempi sillabe"] = [f"{a}-{b}" for a, b in trovate[:5]]   # da controllare: un trattino vero si perderebbe
    return re.sub(r"(\w)-\n([a-zà-ù])", r"\1\2", t)


def sommario(righe, conta):
    """Indice: righe che finiscono con puntini e numero di pagina. Se una pagina è fatta quasi solo di righe così,
    è una pagina di sommario: la tolgo tutta (comprese le voci che vanno a capo, senza puntini)."""
    voce = lambda r: bool(re.search(r"\.{3,}\s*\d+\s*$", r)) or r.strip() == "Sommario"
    tenute, pagina = [], []
    def chiudi():
        piene = [r for r in pagina if r.strip() and not SEGNAPOSTO.match(r)]
        if piene and sum(map(voce, piene)) >= 0.5 * len(piene):
            conta["righe del sommario"] += len(piene)
            tenute.extend(r for r in pagina if SEGNAPOSTO.match(r))       # resta solo il segnaposto della pagina
        else:
            conta["righe del sommario"] += sum(map(voce, piene))
            tenute.extend(r for r in pagina if not voce(r))
    for r in righe:
        if SEGNAPOSTO.match(r) and pagina:
            chiudi(); pagina = []
        pagina.append(r)
    chiudi()
    return tenute


RUMORE_WEB = [r"^Condividi$", r"^-$", r"^Indietro$", r"^#*\s*Questionario di Valutazione$", r"^#+\s*$",
              r"^-?\s*a\.a\. \d{4}/\d{2}$", r"PDF Altri formati$", r"^\|[^|]*\|?\s*$"]


def rumore_web(righe, conta):
    # tutto quello che sta prima del primo titolo «# …» sono briciole di pane e pulsanti
    primo = next((i for i, r in enumerate(righe) if r.startswith("# ")), 0)
    conta["righe prima del titolo (briciole di pane)"] += sum(1 for r in righe[:primo] if r.strip())
    righe = righe[primo:]
    tenute = []
    for r in righe:
        if any(re.search(p, r.strip()) for p in RUMORE_WEB):
            conta["righe di navigazione, link e menu"] += 1
            continue
        doppia = re.match(r"^(.*?) \| (.+?) \| \| \2$", r)       # «Sede | Edificio F | | Edificio F»
        if doppia:
            r = f"{doppia.group(1)} | {doppia.group(2)}"; conta["celle ripetute"] += 1
        tenute.append(r)
    return tenute


# ---------- 2. regole del file ----------
def applica_regola(t, regola):
    """Restituisce (testo, quante volte la regola ha trovato il suo bersaglio)."""
    az = regola["azione"]
    if az == "togli_righe":
        righe = t.split("\n"); tenute = [r for r in righe if not any(re.search(p, r) for p in regola["righe"])]
        return "\n".join(tenute), len(righe) - len(tenute)
    if az == "togli_tra":
        righe = t.split("\n")
        i = next((k for k, r in enumerate(righe) if re.search(regola["da"], r)), None)
        j = next((k for k, r in enumerate(righe) if i is not None and k > i and re.search(regola["a"], r)), None)
        if i is None or j is None: return t, 0
        return "\n".join(righe[:i + 1] + righe[j:]), j - i - 1           # tiene le righe di inizio e di fine, toglie quelle in mezzo
    if az == "togli_frase":
        n = t.count(regola["frase"]); return t.replace(regola["frase"], "").replace("  ", " "), n
    if az == "aggiungi_nota":
        n = t.count(regola["frase"]); return t.replace(regola["frase"], regola["frase"] + " " + regola["nota"]), n
    if az == "sposta_nota":
        righe = t.split("\n")
        i = next((k for k, r in enumerate(righe) if re.search(regola["nota"], r)), None)
        if i is None or regola["richiamo"] not in t: return t, 0
        j = i
        while not righe[j].rstrip().endswith(".") and j + 1 < len(righe): j += 1   # la nota continua fino al punto
        nota = " ".join(r.strip() for r in righe[i:j + 1])
        nota = re.sub(r"^\d+\s+", "", nota)
        t = "\n".join(righe[:i] + righe[j + 1:])
        return t.replace(regola["richiamo"], f" (nota: {nota})", 1), 1
    raise ValueError(f"azione sconosciuta: {az}")


def main():
    PULITO.mkdir(parents=True, exist_ok=True)
    inv = pd.read_excel(INVENTARIO_EXCEL, sheet_name="Inventario", dtype=str).fillna("")
    nel_corpus = [d for d, s in zip(inv["ID"], inv["Nel corpus?"]) if s.strip().lower() == "sì"]
    regole = json.loads(REGOLE_PULIZIA.read_text(encoding="utf-8"))
    rapporto, testi = {}, {}
    for did in nel_corpus:
        sorgente = TESTO / f"{did}.md"
        if not sorgente.exists():
            print(f"{did}: manca {sorgente}"); continue
        testa, corpo = separa_intestazione(sorgente.read_text(encoding="utf-8"))
        conta = Counter(); prima = len(corpo)
        corpo = caratteri(corpo, conta)
        righe = corpo.split("\n")
        if any(SEGNAPOSTO.match(r) for r in righe):                  # PDF
            righe = sommario(righe_ripetute_pdf(righe, conta), conta)
            corpo = sillabe("\n".join(righe), conta)
        else:                                                         # pagina web
            corpo = "\n".join(rumore_web(righe, conta))
        for r in [r for r in regole if r["documento"] == did]:       # le tue decisioni
            corpo, n = applica_regola(corpo, r)
            conta[f"regola «{r['azione']}»: {r['motivo'][:50]}…"] += n
            if n == 0:
                conta["REGOLE SENZA EFFETTO"] += 1
                print(f"  ⚠ {did}: la regola «{r['azione']}» non ha trovato il suo testo. La pagina è cambiata? Motivo della regola: {r['motivo']}")
        corpo = re.sub(r"\n{3,}", "\n\n", corpo).strip() + "\n"
        (PULITO / f"{did}.md").write_text(testa + "\n" + corpo, encoding="utf-8")
        testi[did] = norm(corpo)
        rapporto[did] = {"caratteri prima": prima, "caratteri dopo": len(corpo), **conta}
        print(f"{did}: {prima} → {len(corpo)} caratteri")
        for k, v in conta.items():
            if v and not k.startswith("esempi"): print(f"     {k}: {v}")
        for k in ("esempi ripetute", "esempi sillabe"):
            if conta.get(k): print(f"     {k}: {conta[k]}")
    (CARTELLA / "pulizia.json").write_text(json.dumps(rapporto, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---------- 3. i fatti delle domande di prova ci sono ancora? ----------
    url_a_id = {norm(u): i for i, u in zip(inv["ID"], inv["URL"])}
    dom = pd.read_excel(DOMANDE, sheet_name="Domande", dtype=str).fillna("")
    mancano, controllate = [], 0
    for _, q in dom.iterrows():
        if not q["Fatti verificati"].startswith("sì") or not q["Fatti attesi (separati da |)"]:
            continue
        fonti = [url_a_id.get(norm(q["URL della fonte"]))] if q["URL della fonte"] else [LETTERE[x] for x in re.findall(r"\b[RTPCF]\b", q["Fonte"])]
        fonti = [f for f in fonti if f in testi]
        if not fonti: continue
        controllate += 1
        for f in q["Fatti attesi (separati da |)"].split("|"):
            if f.strip() and not any(norm(f) in testi[x] for x in fonti):
                mancano.append((q["ID"], f.strip(), "/".join(fonti)))
    print(f"\n== Dopo la pulizia ==\nDomande controllate: {controllate}. Fatti persi con la pulizia: {len(mancano)}.")
    for qid, f, fo in mancano:
        print(f"  {qid} [{fo}] «{f}»")
    if mancano:
        print("Un fatto perso vuol dire che la pulizia ha tolto contenuto vero: controlla la regola che l'ha tolto.")


if __name__ == "__main__":
    main()
