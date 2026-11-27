"""
Fermata 6 della linea: ricomporre il testo, dalle righe della pagina ai paragrafi.

Legge  documenti/pulito/Dxx.md     (uscita della fermata 5)
Scrive documenti/ricomposto/Dxx.md

Per i PDF (quaderno B7 passo 3, linea fermata 6):
  1. ogni riga viene riconosciuta: titolo, voce di elenco, nota con asterisco, frase di tabella, testo normale
  2. le righe di testo dello stesso paragrafo si uniscono con uno spazio; un paragrafo finisce quando la riga
     termina con un punto (o due punti, punto e virgola…) ed è più corta di una riga piena
  3. i titoli che vanno a capo («1.6 FASCE … ANNUALE» + «(COA)») si riuniscono
  4. le frasi spezzate dal salto pagina si ricuciono; il segnaposto di pagina si sposta all'inizio del paragrafo
     in cui comincia la pagina, così ogni paragrafo resta associato alla pagina in cui inizia (per le citazioni)
Per le pagine web: i paragrafi arrivano già interi dall'HTML; si uniscono solo le righe spezzate da un «a capo».

Poi il controllo: i fatti delle domande di prova ci sono ancora (il confronto ignora gli a capo, quindi
un fatto perso qui vuol dire testo perso, non solo spostato).

Uso:  python -m unisa.ingestione.ricomposizione
"""
import re

import pandas as pd

from unisa.ingestione.pulizia import norm, separa_intestazione, LETTERE   # stesse funzioni della fermata 5

from unisa.percorsi import DOCUMENTI, DOMANDE, INVENTARIO_EXCEL

CARTELLA = DOCUMENTI
PULITO, RICOMPOSTO = CARTELLA / "pulito", CARTELLA / "ricomposto"
SEGNAPOSTO = re.compile(r"^<!-- Pagina PDF (\d+) -->$")
TITOLO = re.compile(r"^(\d+(\.\d+)+|\d+\.)\s+[A-ZÀ-Ý]")          # «1.5 TASSA…», «3. QUANTO PAGARE»
TABELLA = re.compile(r"^\d+(\.\d+)*\.?\s+[^,]+, .+: .+\.$")        # frasi delle tabelle, scritte alla fermata 4
VOCE = re.compile(r"^(- |o |\*+ ?)")                               # «- …», sottovoci «o …», note «* …»
FINE = re.compile(r"[.:;!?)]$")


def tipo(r):
    s = r.strip()
    if not s: return "vuota"
    if SEGNAPOSTO.match(s): return "pagina"
    if TABELLA.match(s): return "tabella"
    if TITOLO.match(s) or (s.isupper() and len(s) < 80): return "titolo"   # anche «REGOLAMENTO IN MATERIA…»
    if VOCE.match(s): return "voce"
    return "testo"


def ricomponi_pdf(corpo, conta):
    righe = corpo.split("\n")
    # le righe vuote attorno a un segnaposto separano le pagine, non i paragrafi: si tolgono,
    # così una frase che continua sulla pagina dopo si può ricucire
    righe = [r for i, r in enumerate(righe) if r.strip() or not (
        any(SEGNAPOSTO.match(x.strip()) for x in righe[i + 1:i + 2]) or any(SEGNAPOSTO.match(x.strip()) for x in righe[max(0, i - 1):i]))]
    lunghe = sorted(len(r) for r in righe if tipo(r) == "testo")
    piena = lunghe[int(0.9 * len(lunghe))] if lunghe else 100      # lunghezza di una riga piena di questo documento
    paragrafi, cur, pagina_cur, salto = [], None, None, False

    def chiudi():
        nonlocal cur
        if cur:
            paragrafi.append(cur); cur = None

    for r in righe:
        t, s = tipo(r), r.strip()
        if t == "pagina":
            pagina_cur = SEGNAPOSTO.match(s).group(1)
            salto = True                                                   # ricordo che c'è stato un salto pagina
            continue
        if t == "vuota":
            chiudi(); continue
        # un titolo che va a capo: la riga dopo comincia minuscola, con «(», o è tutta maiuscola come il titolo
        if cur and cur["tipo"] == "titolo" and cur["aperto"] and t in ("testo", "titolo") and not TITOLO.match(s) and (s[0].islower() or s[0] == "(" or (s.isupper() and cur["testo"].isupper())):
            cur["testo"] += " " + s; conta["titoli riuniti"] += 1
            cur["aperto"] = not FINE.search(s) and s[0].islower(); continue
        # continuazione di un paragrafo di testo o di una voce di elenco
        # dopo un salto pagina si ricuce solo se la frase è davvero rimasta a metà:
        # una riga che finisce con il punto, seguita da una maiuscola, è la fine di un paragrafo
        if salto and cur and FINE.search(cur["testo"]) and s[:1].isupper():
            cur["aperto"] = False
        if cur and cur["aperto"] and cur["tipo"] in ("testo", "voce") and t == "testo":
            if salto:
                conta["frasi ricucite tra due pagine"] += 1
                conta.setdefault("esempi", []).append(f"…{cur['testo'][-40:]} ⟶ {s[:40]}…")
            cur["testo"] += " " + s; conta["righe unite"] += 1; salto = False
            cur["aperto"] = not (FINE.search(s) and len(s) < 0.8 * piena); continue
        chiudi(); salto = False
        cur = {"tipo": t, "testo": s, "pagina": pagina_cur,
               "aperto": t == "titolo" or (t in ("testo", "voce") and not (FINE.search(s) and len(s) < 0.8 * piena))}
    chiudi()

    out, ultima = [], None
    for p in paragrafi:
        if p["pagina"] != ultima:                     # il segnaposto va davanti al primo paragrafo che comincia su quella pagina
            out.append(f"<!-- Pagina PDF {p['pagina']} -->"); ultima = p["pagina"]
        out.append(p["testo"])
        conta["paragrafi"] += 1
    return "\n\n".join(out) + "\n"


def ricomponi_web(corpo, conta):
    blocchi = re.split(r"\n\s*\n", corpo)
    out = []
    for b in blocchi:
        righe = [r.strip() for r in b.split("\n") if r.strip()]
        if not righe: continue
        if any(r.startswith(("#", "- ")) or " | " in r for r in righe):
            out.extend(righe)                         # titoli, elenchi, righe di tabella: restano righe a sé
        else:
            out.append(" ".join(righe)); conta["righe unite"] += len(righe) - 1
        conta["paragrafi"] += 1
    return "\n\n".join(out) + "\n"


def main():
    from collections import Counter
    RICOMPOSTO.mkdir(parents=True, exist_ok=True)
    testi = {}
    for f in sorted(PULITO.glob("D*.md")):
        testa, corpo = separa_intestazione(f.read_text(encoding="utf-8"))
        conta = Counter()
        pdf = any(SEGNAPOSTO.match(r) for r in corpo.split("\n"))
        nuovo = ricomponi_pdf(corpo, conta) if pdf else ricomponi_web(corpo, conta)
        (RICOMPOSTO / f.name).write_text(testa + "\n" + nuovo, encoding="utf-8")
        testi[f.stem] = norm(nuovo)
        print(f"{f.stem}: {sum(1 for r in corpo.split(chr(10)) if r.strip())} righe → {conta['paragrafi']} paragrafi"
              + "".join(f" · {k}: {v}" for k, v in conta.items() if k not in ("paragrafi", "esempi") and v))
        for e in conta.get("esempi", []): print("     ricucita:", e)

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
            if fa.strip() and not any(norm(fa) in testi[x] for x in fonti):
                mancano.append((q["ID"], fa.strip()))
    print(f"\n== Dopo la ricomposizione ==\nDomande controllate: {controllate}. Fatti persi: {len(mancano)}.")
    for qid, fa in mancano: print(f"  {qid} «{fa}»")
    print("\nGuarda a occhio documenti/ricomposto/D01.md: un paragrafo per blocco, titoli da soli, elenchi voce per voce.")


if __name__ == "__main__":
    main()
