# pricing_agent.py
import os
import re
import json
import time
import argparse
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

CLASS_ID_TO_LABEL = {
    0: "headlamp",
    1: "rear_bumper",
    2: "door",
    3: "hood",
    4: "front_bumper",
}


# ----------------------------
# Utils: currency / price parsing
# ----------------------------
PRICE_RE = re.compile(
    r"""
    (?<!\w)
    (?:€\s*)?
    (?:
        \d{1,3}(?:[ .]\d{3})*(?:[.,]\d{2})?   # 1 234,56 or 1.234,56
        |
        \d+(?:[.,]\d{2})?                    # 1234,56
    )
    \s*(?:€|EUR|euros?)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

BAD_PRICE_HINTS = [
    "mensuel", "/mois", "mois", "leasing", "location", "crédit", "assurance",
    "abonnement", "taux", "taeg"
]

def normalize_price_token(s: str) -> Optional[float]:
    s = s.strip().lower()
    # remove currency words/symbols
    s = s.replace("eur", "").replace("euros", "").replace("euro", "").replace("€", "").strip()

    # remove spaces
    s = s.replace(" ", "")

    # handle thousand separators:
    # - If both '.' and ',' exist, assume the last one is decimal
    if "." in s and "," in s:
        # remove thousand sep: the one that is NOT the last occurrence
        if s.rfind(",") > s.rfind("."):
            # comma is decimal, dot is thousand
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            # dot is decimal, comma is thousand
            s = s.replace(",", "")
    else:
        # only comma -> decimal
        if "," in s:
            s = s.replace(",", ".")
        # only dot -> decimal or thousand; ambiguous. If multiple dots, treat as thousand.
        if s.count(".") > 1:
            s = s.replace(".", "")

    try:
        val = float(s)
        # filter insane values
        if val <= 0:
            return None
        return val
    except Exception:
        return None

def extract_prices_from_text(text: str) -> List[float]:
    t = text.lower()
    if any(h in t for h in BAD_PRICE_HINTS):
        # still allow, but it’s a warning pattern; we won't hard block
        pass

    prices: List[float] = []
    for m in PRICE_RE.finditer(text):
        token = m.group(0)
        val = normalize_price_token(token)
        if val is None:
            continue
        # filter out obviously wrong tiny values like "1,2"
        if val < 5:  # rarely a part price < 5€
            continue
        # filter out very large values (complete car)
        if val > 5000:
            continue
        prices.append(val)
    return prices


# ----------------------------
# SerpApi
# ----------------------------
def serpapi_search(api_key: str, query: str, gl: str = "fr", hl: str = "fr", num: int = 10) -> Dict[str, Any]:
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "gl": gl,
        "hl": hl,
        "num": num,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def collect_candidate_prices(serp: Dict[str, Any]) -> List[Tuple[float, str]]:
    """
    Returns list of (price, source_title_or_domain)
    """
    candidates: List[Tuple[float, str]] = []

    # 1) shopping_results (if present)
    for item in serp.get("shopping_results", []) or []:
        # SerpApi often provides a clean "price" string
        price_str = str(item.get("price", "") or "")
        title = str(item.get("title", "") or "shopping_result")
        prices = extract_prices_from_text(price_str)
        for p in prices:
            candidates.append((p, title))

    # 2) inline images / knowledge graph sometimes embed price snippets
    # (skip for now)

    # 3) organic_results snippets
    for item in serp.get("organic_results", []) or []:
        title = str(item.get("title", "") or "organic_result")
        snippet = str(item.get("snippet", "") or "")
        link = str(item.get("link", "") or "")
        domain = link.split("/")[2] if "://" in link else link
        src = f"{title} ({domain})" if domain else title

        # Search prices in snippet + title
        text = f"{title}\n{snippet}"
        for p in extract_prices_from_text(text):
            candidates.append((p, src))

    return candidates


# ----------------------------
# Preds.json parsing
# ----------------------------
def load_preds(preds_path: str) -> List[Dict[str, Any]]:
    with open(preds_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # support multiple shapes
    if isinstance(data, dict):
        # maybe {"predictions":[...]} or {"instances":[...]}
        for k in ("preds", "predictions", "instances", "outputs"):
            if k in data and isinstance(data[k], list):
                return data[k]
        # maybe single prediction dict
        return [data]

    if isinstance(data, list):
        return data

    raise ValueError("preds.json format inconnu")


def get_label_from_pred(p):
    if "label" in p and p["label"]:
        return str(p["label"])

    if "cls_id" in p:
        return CLASS_ID_TO_LABEL.get(int(p["cls_id"]), "unknown")

    if "class_id" in p:
        return CLASS_ID_TO_LABEL.get(int(p["class_id"]), "unknown")

    return "unknown"



def get_score_from_pred(p: Dict[str, Any]) -> Optional[float]:
    for k in ("score", "confidence", "prob"):
        if k in p and p[k] is not None:
            try:
                return float(p[k])
            except Exception:
                pass
    return None


def unique_parts(preds: List[Dict[str, Any]], min_score: float = 0.5) -> List[Dict[str, Any]]:
    parts: Dict[str, Dict[str, Any]] = {}
    for p in preds:
        label = get_label_from_pred(p)
        score = get_score_from_pred(p)
        if score is not None and score < min_score:
            continue

        # keep best score per label
        if label not in parts or (score is not None and (parts[label].get("score") or 0) < score):
            parts[label] = {"label": label, "score": score}

    # keep stable order
    return list(parts.values())


# ----------------------------
# Query builder
# ----------------------------
# OPTIONAL: map your dataset labels to nicer French queries
LABEL_TO_QUERY = {
    "door": "porte",
    "front_bumper": "pare-chocs avant",
    "rear_bumper": "pare-chocs arrière",
    "hood": "capot",
    "headlamp": "phare avant",
    # if your detectron outputs class_1, class_2... you can map them too
    # "class_1": "phare avant", etc.
}

def build_query(make: str, model: str, year: str, part_label: str) -> str:
    part_q = LABEL_TO_QUERY.get(part_label, part_label.replace("_", " "))
    # We include key hints: "prix", "pièce", "OEM" to bias results
    return f'prix {part_q} {make} {model} {year} pièce auto'


# ----------------------------
# Main pricing logic
# ----------------------------
def estimate_price_for_part(api_key: str, make: str, model: str, year: str, part_label: str) -> Dict[str, Any]:
    query = build_query(make, model, year, part_label)
    serp = serpapi_search(api_key, query=query, gl="fr", hl="fr", num=10)
    candidates = collect_candidate_prices(serp)

    values = [p for p, _ in candidates]
    # robust estimate: use median if enough values
    result: Dict[str, Any] = {
        "part_label": part_label,
        "query": query,
        "num_candidates": len(values),
        "estimate": None,
        "min": None,
        "max": None,
        "median": None,
        "samples": [],
    }

    # keep up to 10 sample sources
    for p, src in candidates[:10]:
        result["samples"].append({"price": p, "source": src})

    if len(values) >= 3:
        result["min"] = round(min(values), 2)
        result["max"] = round(max(values), 2)
        result["median"] = round(median(values), 2)
        result["estimate"] = result["median"]
    elif len(values) > 0:
        # fallback: average of available values
        avg = sum(values) / len(values)
        result["min"] = round(min(values), 2)
        result["max"] = round(max(values), 2)
        result["estimate"] = round(avg, 2)

    return result


def main():
    load_dotenv()
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ SERPAPI_API_KEY manquante dans .env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds.json", help="Chemin vers preds.json (sortie infer.py)")
    ap.add_argument("--make", required=True, help="Marque (ex: Volkswagen)")
    ap.add_argument("--model", required=True, help="Modèle (ex: Golf)")
    ap.add_argument("--year", required=True, help="Année (ex: 2019)")
    ap.add_argument("--min-score", type=float, default=0.5, help="Seuil score detection (default 0.5)")
    ap.add_argument("--sleep", type=float, default=1.0, help="Pause entre requêtes (évite rate-limit)")
    ap.add_argument("--out", default="pricing_report.json", help="Fichier de sortie")
    args = ap.parse_args()

    preds = load_preds(args.preds)
    parts = unique_parts(preds, min_score=args.min_score)

    report = {
        "car": {"make": args.make, "model": args.model, "year": args.year},
        "preds_file": args.preds,
        "min_score": args.min_score,
        "parts_detected": parts,
        "pricing": [],
        "total_estimate": None,
        "currency": "EUR",
    }

    total = 0.0
    for i, part in enumerate(parts, start=1):
        label = part["label"]
        print(f"[{i}/{len(parts)}] Recherche prix pour: {label} ...")
        info = estimate_price_for_part(api_key, args.make, args.model, args.year, label)
        report["pricing"].append(info)

        if info.get("estimate") is not None:
            total += float(info["estimate"])

        time.sleep(args.sleep)

    report["total_estimate"] = round(total, 2) if report["pricing"] else None

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n✅ Terminé")
    print(f"→ Report: {args.out}")
    if report["total_estimate"] is not None:
        print(f"→ Total estimé: {report['total_estimate']} EUR")


if __name__ == "__main__":
    main()
