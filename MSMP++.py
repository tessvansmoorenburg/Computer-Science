import json
import re
import time
import random
import os
from typing import List, Dict, Set, Any, Tuple
from itertools import combinations
from collections import defaultdict, Counter
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.interpolate import PchipInterpolator
from bayes_opt import BayesianOptimization

# ==============================================================================
# 1. CONFIGURATION & GLOBAL CONSTANTS
# ==============================================================================

# --- Random Seeding for Reproducibility ---
RND = 42
random.seed(RND)
np.random.seed(RND)
PRIME = 2_147_483_647  # Mersenne prime for MinHash

# --- MSM Hyperparameters optimized (Globals - will be overwritten dynamically) ---
EPSILON = 0.452 
MU = 0.8
ALPHA = 0.572
BETA = 0.0
GAMMA = 0.9

# --- Bootstrap & Performance Settings ---
NUM_BOOTSTRAPS = 5
PRUNING_THRESHOLD = 0.02  # Frequency threshold for stop word identification

# --- Regex Patterns ---
# Matches alphanumeric strings containing at least one digit
MODELWORD_TITLE_RE = re.compile(r"[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*")
# Matches specific value patterns (e.g., "1080p", "55inch")
MODELWORD_VAL_RE = re.compile(r"^\d+(\.\d+)?([a-zA-Z]+)?$")

# --- Data Cleaning: Trash/Stop Words ---
TRASH_WORDS = {
    'hdtv', '1080p', 'class', 'led', 'diag', 'newegg', 'com', 'best', 'buy', 'lcd',
    '3d', 'tv', '60hz', '120hz', 'smart', '720p', 'plasma', 'refurbished', 'inch',
    '240hz', '600hz', 'black', 'ledlcd', 'model', 'hz', 'diagonal', 'viera', 'tc',
    'with', 'hd', 'aquos', 'slim', 'series', 'bravia', 'cinema', 'lc', 'full', 'size',
    'and', 'net', 'ultra', 'widescreen', 'trumotion', 'dvd', 'rb', 'clearscan', 'ns',
    'pairs', 'glasses', 'electronics', 'of', 'internet', 'silver', 'kdl', 'wifi',
    'wi-fi', 'built-in', 'bluetooth', 'lan', 'ethernet', 'hdmi', 'usb', 'input',
    'output', 'port', 'ports', 'slot', 'watt', 'volt', 'v', 'w', 'mw', 'khz', 'mhz',
    'ghz', 'gb', 'mb', 'tb', 'kg', 'lbs', 'lb', 'oz', 'cm', 'mm', 'm', 'in', 'inches',
    'dolby', 'digital', 'surround', 'audio', 'video', 'sound', 'stereo', 'speaker',
    'speakers', 'display', 'panel', 'screen', 'monitor', 'type', 'version', 'ver',
    'hdr', 'hdr10', 'active', 'passive', 'definition', 'resolution', 'remote',
    'control', 'stand', 'mount', 'wall', 'bracket', 'cable', 'cord', 'power',
    'adapter', 'box', 'gray', 'grey', 'white', 'color', 'colour', 'thin', 'curved',
    'flat', 'sale', 'deal', 'offer', 'cheap', 'discount', 'warranty', 'year',
    'official', 'authentic', 'usa', 'international', 'global', 'new', 'used',
    'renewed', 'open', 'warehouse', 'for', 'the', 'a', 'an', 'by', 'from', 'to',
    'in', 'on', 'at', 'or', 'w/', 'without', 'is', 'are'
}


# ==============================================================================
# 2. TEXT PROCESSING & EXTRACTION HELPERS
# ==============================================================================

def unit_normalize(s: str) -> str:
    """Normalizes units in a string (e.g., 'inches' -> 'inch', 'hertz' -> 'hz')."""
    if s is None:
        return ""
    
    s = str(s).lower().replace(",", "").replace(";", "").replace(":", "").replace("/", " ")
    
    unit_synonym_map = {
        "inches": "inch", " inch": "inch", "-inch": "inch", "”": "inch", "'": "inch",
        "centimeter": "cm", "millimeters": "mm", "mm": "mm", "meter": "m",
        "hertz": "hz", "kilohertz": "khz", "megahertz": "mhz", "gigahertz": "ghz",
        "kilograms": "kg", "pounds": "lb", "lbs": "lb", "kilogram": "kg",
        "gigabytes": "gb", "megabytes": "mb", "terabytes": "tb",
    }
    
    for long_form, short_form in unit_synonym_map.items():
        s = s.replace(long_form, short_form)
        
    # Merge numbers with units (e.g., "55 inch" -> "55inch")
    units_to_merge = "(inch|hz|khz|mhz|ghz|cm|mm|m|kg|lb|mb|gb|tb)"
    s = re.sub(r"(\d+(\.\d+)?)\s*(" + units_to_merge + r")", r"\1\3", s)
    return s


def normalize_text(s: Any) -> str:
    """Basic lowercasing and punctuation removal."""
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9\.\-\_\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_brand(text: str) -> str:
    """Extracts and normalizes brand names from text against a known list."""
    brand_map = {
        "samsung": "samsung", "sony": "sony", "lg": "lg", "vizio": "vizio",
        "panasonic": "panasonic", "philips": "philips", "toshiba": "toshiba",
        "sharp": "sharp", "jvc": "jvc", "sanyo": "sanyo", "rca": "rca",
        "hisense": "hisense", "tcl": "tcl", "insignia": "insignia",
        "sceptre": "sceptre", "westinghouse": "westinghouse", "seiki": "seiki",
        "element": "element", "haier": "haier", "upstar": "upstar",
        "viewsonic": "viewsonic", "nec": "nec", "supersonic": "supersonic",
        "hannspree": "hannspree", "proscan": "proscan", "epson": "epson",
        "sunbrite": "sunbrite", "pyle": "pyle", "sansui": "sansui"
    }
    
    if not text:
        return "__unknown__"
    
    s = normalize_text(text)
    for key, val in brand_map.items():
        if key in s:
            return val
    return "__unknown__"


def extract_model_words_generic(text: str) -> Set[str]:
    """Base function to extract alphanumeric model identifiers using Regex."""
    if not text:
        return set()
    
    tokens = set()
    # Strategy 1: Alphanumeric mix
    for m in MODELWORD_TITLE_RE.finditer(text.lower()):
        tokens.add(m.group(0))
    
    # Strategy 2: Numeric values with logical constraints
    for m in MODELWORD_VAL_RE.finditer(text.lower()):
        w = m.group(0)
        # Filter out small single integers often used for quantity
        if not w.isdigit() or int(w) > 5:
            tokens.add(w)
            
    return {t for t in tokens if t not in TRASH_WORDS}


def extract_title_model_words(text: str) -> Set[str]:
    """Wrapper to extract model words specifically from titles."""
    return extract_model_words_generic(text)


def extract_value_model_words(text: str) -> Set[str]:
    """Wrapper to extract model words from attribute values (includes unit normalization)."""
    text = unit_normalize(text)
    return extract_model_words_generic(text)


def extract_title_non_model_words(text: str) -> Set[str]:
    """Extracts clean tokens from titles that are NOT model words."""
    if not text:
        return set()
    
    text = unit_normalize(text)
    text = normalize_text(text)
    all_tokens = set(text.split())
    mw_tokens = extract_title_model_words(text)
    
    clean_tokens = (all_tokens - mw_tokens) - TRASH_WORDS
    return {t for t in clean_tokens if not t.isdigit()}


# ==============================================================================
# 3. DATA LOADING & FEATURE ENGINEERING
# ==============================================================================

def load_products_from_json(json_path: str) -> List[Dict]:
    """Loads and parses the raw JSON dataset into a structured list."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[ERROR] Loading JSON: {e}")
        return []

    products = []
    if isinstance(raw, dict):
        for modelid, items in raw.items():
            for it in items:
                raw_title = it.get("title", "")
                cleaned_title = normalize_text(unit_normalize(raw_title))
                brand = normalize_brand(raw_title)
                
                # Process features map
                attrs = {}
                for k, v in (it.get("featuresMap") or {}).items():
                    norm_v = normalize_text(unit_normalize(str(v)))
                    attrs[k] = norm_v
                
                products.append({
                    "shop": it.get("shop", "").lower(),
                    "title": cleaned_title,
                    "brand": brand,
                    "model": str(modelid).lower(),
                    "attributes": attrs,
                })
    return products


def get_product_features(product: Dict[str, Any], method_name: str, stop_words: Set[str] = None) -> List[str]:
    """
    Extracts a list of string features for LSH from a product.
    Supports 'msmp' (standard) and 'msmp++' (with stop word pruning).
    """
    if stop_words is None:
        stop_words = set()
        
    features = []
    mw_set = extract_title_model_words(product['title'])
    
    if product['brand'] != "__unknown__":
        features.append(product['brand'])
    
    # Include attribute model words if using MSMP methods
    if 'msmp' in method_name:
        all_values = " ".join(str(v) for v in product.get("attributes", {}).values())
        mw_set.update(extract_value_model_words(all_values))
    
    # Apply stop word removal for MSMP++
    if method_name == 'msmp++':
        mw_set = mw_set - stop_words
        
    features.extend(list(mw_set))
    return features


def get_global_token_counts(products: List[Dict]) -> Counter:
    """Counts occurrence of all tokens across the entire dataset."""
    token_counts = Counter()
    for p in products:
        features = set()
        features.update(extract_title_model_words(p['title']))
        all_values = " ".join(str(v) for v in p.get("attributes", {}).values())
        features.update(extract_value_model_words(all_values))
        token_counts.update(features)
    return token_counts


def identify_stop_words(token_counts: Counter, total_products: int, threshold: float) -> Set[str]:
    """Identifies tokens that appear too frequently (based on threshold)."""
    limit = total_products * threshold
    return {t for t, count in token_counts.items() if count > limit}


def build_vocab_index(products: List[Dict], method_name: str, stop_words: Set[str] = None) -> Dict[str, int]:
    """Creates a mapping from feature string to integer index."""
    vocab = set()
    for p in products:
        vocab.update(get_product_features(p, method_name, stop_words))
    vocab_list = sorted(list(vocab))
    return {feat: i for i, feat in enumerate(vocab_list)}


def build_sparse_matrix(products: List[Dict], method_name: str, vocab_index: Dict[str, int], stop_words: Set[str] = None) -> List[List[int]]:
    """Converts products into a sparse representation (list of token indices)."""
    sparse_matrix = []
    for p in products:
        feats = get_product_features(p, method_name, stop_words)
        indices = [vocab_index[f] for f in feats if f in vocab_index]
        sparse_matrix.append(indices)
    return sparse_matrix


def find_optimal_nsig_dynamic(n_features: int) -> int:
    """Heuristic to determine minhash signature length."""
    target = int(max(10, n_features * 0.5))
    return max(10, target)


# ==============================================================================
# 4. LSH (LOCALITY SENSITIVE HASHING) & MINHASH
# ==============================================================================

def compute_minhash_signatures(sparse_matrix: List[List[int]], n_sig: int) -> np.ndarray:
    """
    Computes the MinHash signature matrix for the dataset.
    Uses vectorization for speed.
    """
    N = len(sparse_matrix)
    rng = np.random.RandomState(42)
    
    # Generate random hash functions (a*x + b) % PRIME
    a = rng.randint(1, PRIME, size=(1, n_sig)).astype(np.int64)
    b = rng.randint(0, PRIME, size=(1, n_sig)).astype(np.int64)
    
    sigs = np.full((N, n_sig), PRIME, dtype=np.int64)
    
    for i in range(N):
        indices = sparse_matrix[i]
        if not indices:
            continue
        idx_arr = np.array(indices, dtype=np.int64).reshape(-1, 1)
        # Apply all hash functions to all indices of this document at once
        hashed = (idx_arr * a + b) % PRIME
        # MinHash: Take the minimum hash value for each function
        sigs[i] = hashed.min(axis=0)
        
    return sigs


def lsh_banding(signatures: np.ndarray, r: int) -> Set[Tuple[int, int]]:
    """
    Performs LSH Banding to find candidate pairs.
    r: Number of rows per band.
    """
    N, n_sig = signatures.shape
    b = max(1, n_sig // r)  # Number of bands
    pairs = set()
    
    for band_idx in range(b):
        start = band_idx * r
        end = min(start + r, n_sig)
        
        # Bucket products that have identical signatures in this band
        buckets = defaultdict(list)
        band_sigs = signatures[:, start:end]
        
        for pid, sig_row in enumerate(band_sigs):
            buckets[tuple(sig_row)].append(pid)
            
        for group in buckets.values():
            if len(group) > 1:
                group.sort()
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        pairs.add((group[i], group[j]))
    return pairs


# ==============================================================================
# 5. MSM (MULTI-COMPONENT SIMILARITY METHOD)
# ==============================================================================

@lru_cache(maxsize=100000)
def qgrams(s: str, q: int = 3) -> List[str]:
    """Generates q-grams for a string."""
    ss = "#" + s + "#"
    return [ss[i:i+q] for i in range(len(ss)-q+1)]


@lru_cache(maxsize=100000)
def qgram_sim(a: str, b: str, q: int = 3) -> float:
    """Calculates Jaccard similarity based on q-grams."""
    if a == "" and b == "": return 1.0
    if a == "" or b == "": return 0.0
    
    ca = Counter(qgrams(a, q))
    cb = Counter(qgrams(b, q))
    
    inter = sum((ca & cb).values())
    union = sum(ca.values()) + sum(cb.values()) - inter
    
    return inter / union if union > 0 else 0.0


def compute_msm_similarity_unpacked(args: Tuple) -> Tuple[int, int, float]:
    """
    Core MSM similarity logic.
    Calculates weighted similarity based on Attributes, Model Words, and Title tokens.
    """
    p1, p2, u, v = args
    
    # --- 1. Key-Value Pair (Attribute) Similarity ---
    kvp_i = list(p1.get("attributes", {}).items())
    kvp_j = list(p2.get("attributes", {}).items())
    
    matched_i, matched_j = set(), set()
    sum_w_sim, w_sum, m = 0.0, 0.0, 0
    potential_matches = []
    
    # Find best matching keys
    for i, (ki, vi) in enumerate(kvp_i):
        for j, (kj, vj) in enumerate(kvp_j):
            key_sim = qgram_sim(ki, kj)
            if key_sim > GAMMA:
                potential_matches.append((key_sim, i, j))
    
    potential_matches.sort(key=lambda x: x[0], reverse=True)
    
    # Calculate weighted value similarity for matched keys
    for score, i, j in potential_matches:
        if i not in matched_i and j not in matched_j:
            matched_i.add(i)
            matched_j.add(j)
            val_sim = qgram_sim(kvp_i[i][1], kvp_j[j][1])
            sum_w_sim += score * val_sim
            w_sum += score
            m += 1
            
    avgSim = sum_w_sim / w_sum if w_sum > 0 else 0.0
    
    # --- 2. Model Word (MW) Percentage from Unmatched Attributes ---
    nmk_i = " ".join([v for i, (k, v) in enumerate(kvp_i) if i not in matched_i])
    nmk_j = " ".join([v for j, (k, v) in enumerate(kvp_j) if j not in matched_j])
    
    mw_i = extract_value_model_words(nmk_i)
    mw_j = extract_value_model_words(nmk_j)
    
    mwPerc = len(mw_i & mw_j) / len(mw_i | mw_j) if (mw_i or mw_j) else 0.0
    
    # --- 3. Title Model Word Similarity ---
    t_mw1 = extract_title_model_words(p1['title'])
    t_mw2 = extract_title_model_words(p2['title'])
    title_mw_sim = 0.0
    
    if t_mw1 and t_mw2:
        # Best match average
        s1 = [max([qgram_sim(w1, w2) for w2 in t_mw2]) for w1 in t_mw1]
        s2 = [max([qgram_sim(w1, w2) for w1 in t_mw1]) for w2 in t_mw2]
        title_mw_sim = (np.mean(s1) + np.mean(s2)) / 2.0
        
    # --- 4. Title Non-Model Word Similarity ---
    nmw1 = extract_title_non_model_words(p1['title'])
    nmw2 = extract_title_non_model_words(p2['title'])
    title_token_sim = qgram_sim(" ".join(sorted(nmw1)), " ".join(sorted(nmw2)))

    # --- 5. Final Weighted Aggregation ---
    ratio = m / min(len(kvp_i), len(kvp_j)) if kvp_i and kvp_j else 0.0
    
    if title_mw_sim < ALPHA:
        # If titles don't match well on model words, rely on attributes
        hSim = ratio * avgSim + (1 - ratio) * mwPerc
    else:
        # Otherwise, mix all components
        rem = 1.0 - (MU + BETA)
        hSim = (rem * ratio * avgSim) + (rem * (1 - ratio) * mwPerc) + (MU * title_mw_sim) + (BETA * title_token_sim)
        
    return u, v, max(0.0, min(1.0, hSim))


# ==============================================================================
# 6. EVALUATION METRICS & LOOP
# ==============================================================================

def compute_gold_pairs(products: List[Dict]) -> Set[Tuple[int, int]]:
    """Generates the set of true duplicate pairs based on Model ID."""
    model_map = defaultdict(list)
    for idx, p in enumerate(products):
        model_map[p.get("model", "")].append((idx, p.get("shop")))
        
    gold = set()
    for items in model_map.values():
        if len(items) > 1:
            # Pair products with same Model ID but different Shops
            for (i, s1), (j, s2) in combinations(items, 2):
                if s1 != s2: 
                    gold.add(tuple(sorted((i, j))))
    return gold


def calculate_metrics(candidate_pairs: Set, pred_pairs: Set, gold_pairs: Set) -> Dict[str, float]:
    """Calculates Precision, Recall, F1, PQ (Pair Quality), and PC (Pair Completeness)."""
    gold_set = set(gold_pairs)
    pred_set = set(pred_pairs)
    cand_set = set(candidate_pairs)
    
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # LSH specific metrics
    Df = len(cand_set & gold_set)
    Dn = len(gold_set)
    Nc = len(cand_set)
    
    pq = Df / Nc if Nc > 0 else 0.0
    pc = Df / Dn if Dn > 0 else 0.0
    f1_star = 2 * (pq * pc) / (pq + pc) if (pq + pc) > 0 else 0.0
    
    return {"F1": f1, "F1_star": f1_star, "PQ": pq, "PC": pc}


def run_evaluation_loop_optimized(products: List[Dict], signatures: np.ndarray, forced_n_sig: int) -> List[Dict]:
    """
    Runs the full entity resolution pipeline for various LSH parameters (rows 'r').
    1. LSH Banding -> Candidates
    2. MSM Similarity -> Distance Matrix
    3. Hierarchical Clustering -> Predicted Pairs
    4. Metrics Calculation
    """
    N = len(products)
    total_possible = N * (N - 1) / 2
    gold_pairs = compute_gold_pairs(products)
    agg_metrics = []
    
    # Determine 'r' values to scan (dense low range, sparse high range)
    r_dense = list(range(1, min(21, forced_n_sig + 1)))
    r_sparse = []
    if forced_n_sig > 20: 
        r_sparse = np.unique(np.linspace(21, forced_n_sig, num=15, dtype=int)).tolist()
    
    r_values_to_check = sorted(list(set(r_dense + r_sparse)))
    if 1 not in r_values_to_check:
        r_values_to_check.insert(0, 1)

    for r in r_values_to_check:
        candidates = lsh_banding(signatures, r)
        num_cands = len(candidates)
        
        if num_cands == 0:
            continue

        foc = num_cands / total_possible if total_possible > 0 else 0
        
        # Prepare comparison tasks
        tasks = []
        for u, v in candidates:
            p1, p2 = products[u], products[v]
            # Basic filtering: Different shops, same brand (if known)
            if p1['shop'] == p2['shop']: continue
            if p1['brand'] != "__unknown__" and p2['brand'] != "__unknown__" and p1['brand'] != p2['brand']: continue
            tasks.append((p1, p2, u, v))
            
        # Build distance matrix (default to 1.0 distance)
        D = np.ones((N, N))
        np.fill_diagonal(D, 0.0)
        
        # Compute MSM similarities
        for args in tasks:
            u_out, v_out, sim = compute_msm_similarity_unpacked(args)
            D[u_out, v_out] = D[v_out, u_out] = 1.0 - sim

        # Clustering
        model = AgglomerativeClustering(
            n_clusters=None, metric='precomputed', linkage='complete', 
            distance_threshold=1.0 - EPSILON
        )
        clustering_res = model.fit_predict(D)
        
        # Extract predicted pairs from clusters
        clusters = defaultdict(list)
        for i, label in enumerate(clustering_res):
            clusters[label].append(i)
            
        pred_pairs = set()
        for nodes in clusters.values():
            if len(nodes) > 1:
                for u, v in combinations(nodes, 2):
                    pred_pairs.add(tuple(sorted((u, v))))
        
        met = calculate_metrics(candidates, pred_pairs, gold_pairs)
        met['frac'] = foc
        agg_metrics.append(met)
            
    agg_metrics.sort(key=lambda x: x['frac'])
    return agg_metrics


def run_bootstrap_optimized(products: List[Dict], forced_n_sig: int, vocab_index: Dict, method: str, stop_words: Set = None) -> List[Dict]:
    """Runs the evaluation loop multiple times with bootstrapped samples to smooth results."""
    print(f"   [Setup] Pre-computing signatures for {len(products)} products...")
    sparse_matrix = build_sparse_matrix(products, method, vocab_index, stop_words)
    full_signatures = compute_minhash_signatures(sparse_matrix, forced_n_sig)
    
    # Group indices by model for stratified sampling
    model_groups = defaultdict(list)
    for idx, p in enumerate(products):
        model_groups[p['model']].append(idx)
    unique_models = list(model_groups.keys())
    
    all_curves = []
    print(f"   [Bootstrap] Starting {NUM_BOOTSTRAPS} iterations...")
    
    for k in range(NUM_BOOTSTRAPS):
        rng = np.random.RandomState(RND + k)
        selected_models = rng.choice(unique_models, size=len(unique_models), replace=True)
        
        bootstrap_indices = []
        for mid in selected_models:
            bootstrap_indices.extend(model_groups[mid])
            
        current_products = [products[i] for i in bootstrap_indices]
        current_signatures = full_signatures[bootstrap_indices]
        
        t0 = time.time()
        curve = run_evaluation_loop_optimized(current_products, current_signatures, forced_n_sig)
        t1 = time.time()
        print(f"     Batch {k+1}/{NUM_BOOTSTRAPS} done in {t1-t0:.1f}s")
        all_curves.append(curve)
    
    # Average the curves
    avg_curve = []
    if not all_curves: return []
    min_len = min(len(c) for c in all_curves)
    
    for i in range(min_len):
        step_avg = {}
        for key in ['frac', 'F1', 'F1_star', 'PQ', 'PC']:
            step_avg[key] = np.mean([c[i][key] for c in all_curves])
        avg_curve.append(step_avg)
        
    return avg_curve


# ==============================================================================
# 7. HYPERPARAMETER OPTIMIZATION
# ==============================================================================

class MSMHyperparameterOptimizer:
    """Wrapper class to optimize MSM weights using Bayesian Optimization."""
    
    def __init__(self, products: List[Dict], vocab_index: Dict, n_sig: int, stop_words: Set = None, method: str = 'msmp++'):
        self.products = products
        self.vocab_index = vocab_index
        self.n_sig = n_sig
        self.stop_words = stop_words
        self.method = method # Store method name to select sparse matrix strategy
        
        print(f"[Opt] Pre-computing signatures for optimization ({self.method})...")
        # Ensure we build the sparse matrix matching the method we are tuning
        sparse_matrix = build_sparse_matrix(self.products, self.method, self.vocab_index, self.stop_words)
        self.signatures = compute_minhash_signatures(sparse_matrix, self.n_sig)

    def objective_function(self, epsilon, mu, alpha, beta, gamma):
        """The function Bayesian Optimization tries to maximize (Best F1 score)."""
        global EPSILON, MU, ALPHA, BETA, GAMMA
        
        # Assign trial values to globals
        EPSILON = epsilon
        ALPHA = alpha
        GAMMA = gamma
        
        # Normalize MU and BETA to ensure sum <= 1.0 (approximated)
        total_weight = mu + beta
        if total_weight > 0.99:
            scale = 0.99 / total_weight
            MU = mu * scale
            BETA = beta * scale
        else:
            MU = mu
            BETA = beta

        metrics_list = run_evaluation_loop_optimized(self.products, self.signatures, self.n_sig)
        
        if not metrics_list:
            return 0.0

        # Return the best F1 found across all 'r' settings
        best_f1 = max([m['F1'] for m in metrics_list])
        return best_f1

    def optimize(self, init_points=5, n_iter=25):
        """Runs the optimization process."""
        pbounds = {
            'epsilon': (0.3, 0.7), 
            'mu':      (0.4, 0.8), 
            'alpha':   (0.4, 0.8), 
            'beta':    (0.0, 0.3), 
            'gamma':   (0.6, 0.9), 
        }

        optimizer = BayesianOptimization(
            f=self.objective_function,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )

        print(f"\n>>> Starting Bayesian Optimization ({init_points + n_iter} total runs)...")
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        return optimizer


# ==============================================================================
# 8. PLOTTING UTILITIES
# ==============================================================================

def calculate_auc(data: List[Dict], x_key='frac', y_key='F1') -> float:
    """Calculates Area Under Curve using Trapezoidal rule."""
    x = [d[x_key] for d in data]
    y = [d[y_key] for d in data]
    if len(x) < 2: return 0.0
    return np.trapz(y, x)


def extend_curve(data: List[Dict]) -> List[Dict]:
    """Extrapolates the curve to FOC=1.0 for visualization consistency."""
    if not data: return data
    data = sorted(data, key=lambda x: x['frac'])
    last_point = data[-1]
    if last_point['frac'] < 1.0:
        new_point = last_point.copy()
        new_point['frac'] = 1.0 
        data.append(new_point)
    return data


def plot_results(baseline_data: List[Dict], improved_data: List[Dict]):
    """Generates comparison plots for PQ, PC, F1, and F1*."""
    baseline_data = extend_curve(baseline_data)
    improved_data = extend_curve(improved_data)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Optimization Results: MSMP+ vs MSMP++', fontsize=16)
    
    def get_smooth_xy(data, key):
        x = np.array([d['frac'] for d in data])
        y = np.array([d[key] for d in data])
        
        # Sort and unique
        _, unique_indices = np.unique(x, return_index=True)
        x = x[unique_indices]
        y = y[unique_indices]
        
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        y = y[sorted_indices]
        
        # PCHIP interpolation for smoothness
        if len(x) < 4: return x, y
            
        x_new = np.linspace(x.min(), x.max(), 300)
        try:
            spl = PchipInterpolator(x, y)
            y_smooth = spl(x_new)
            y_smooth = np.clip(y_smooth, 0, 1)
            return x_new, y_smooth
        except:
            return x, y

    metrics = [
        ('PQ', 'Pair Quality (PQ)', axes[0, 0]),
        ('PC', 'Pair Completeness (PC)', axes[0, 1]),
        ('F1', 'F1 Measure', axes[1, 0]),
        ('F1_star', 'F1* Measure (LSH Quality)', axes[1, 1])
    ]

    for key, title, ax in metrics:
        bx, by = get_smooth_xy(baseline_data, key)
        ix, iy = get_smooth_xy(improved_data, key)
        
        ax.plot(bx, by, 'r-', label='MSMP+ (Standard)', linewidth=2.5)
        ax.plot(ix, iy, 'b-', label='MSMP++ (Optimized)', linewidth=2.5)
        
        ax.set_title(title)
        ax.set_xlabel('Fraction of Comparisons (FOC)')
        ax.set_ylabel(key)
        ax.set_xlim(-0.05, 1.05) 
        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('comparison_result.png')
    print("\n[INFO] Plot saved to 'comparison_result.png'")


# ==============================================================================
# 9. MAIN EXECUTION
# ==============================================================================

def set_global_params(params_dict):
    """Updates global hyperparameters from a dictionary."""
    global EPSILON, MU, ALPHA, BETA, GAMMA
    EPSILON = params_dict['epsilon']
    ALPHA = params_dict['alpha']
    GAMMA = params_dict['gamma']
    
    raw_mu = params_dict['mu']
    raw_beta = params_dict['beta']
    
    if raw_mu + raw_beta > 0.99:
        scale = 0.99 / (raw_mu + raw_beta)
        MU = raw_mu * scale
        BETA = raw_beta * scale
    else:
        MU = raw_mu
        BETA = raw_beta

if __name__ == "__main__":
    # >>>>> CONFIGURATION: SET YOUR DATA PATH HERE <<<<<
    JSON_PATH = "TVs-all-merged.json"  
    
    print(">>> Loading Data...")
    products = load_products_from_json(JSON_PATH)
    
    if products:
        # --- 1. PRE-PROCESSING ---
        print(">>> Analyzing Token Frequencies...")
        global_counts = get_global_token_counts(products)
        freq_stop_words = identify_stop_words(global_counts, len(products), threshold=PRUNING_THRESHOLD)
        print(f"   Pruning {len(freq_stop_words)} generic tokens.")

        print(">>> Building Vocabulary...")
        vocab_index = build_vocab_index(products, 'msmp+', stop_words=None)
        FIXED_N_SIG = find_optimal_nsig_dynamic(len(vocab_index))
        print(f"   Vocab: {len(vocab_index)} | n_sig: {FIXED_N_SIG}")
        
        print("\n" + "="*50)
        print(" OPTIMIZING HYPERPARAMETERS")
        print("="*50)
        
        optimizer_prop = MSMHyperparameterOptimizer(
            products, vocab_index, FIXED_N_SIG, stop_words=freq_stop_words, method='msmp++'
        )
        res_prop = optimizer_prop.optimize(init_points=5, n_iter=20)
        best_params = res_prop.max['params']

        # Set globals to baseline best
        set_global_params(best_params)
        print(f"\n>>> Running Baseline Evaluation with BEST Params: Eps={EPSILON:.3f}")
        
        # Run Evaluation for Baseline
        curve_baseline = run_bootstrap_optimized(
            products, FIXED_N_SIG, vocab_index, method='msmp+', stop_words=None
        )

        print(f"\n>>> Running Proposed Evaluation with BEST Params: Eps={EPSILON:.3f}")

        # Run Evaluation for Proposed
        curve_proposed = run_bootstrap_optimized(
            products, FIXED_N_SIG, vocab_index, method='msmp++', stop_words=freq_stop_words
        )
        
        # --- DEBUG OUTPUT (RAW VALUES) ---
        print("\n" + "="*80)
        print(" RAW DATA DEBUG VIEW (MSMP+ Baseline)")
        print("="*80)
        print(f"{'FOC (frac)':<15} | {'F1':<10} | {'F1*':<10} | {'PC':<10} | {'PQ':<10}")
        print("-" * 65)
        for row in curve_baseline:
            print(f"{row['frac']:<15.6f} | {row['F1']:<10.4f} | {row['F1_star']:<10.4f} | {row['PC']:<10.4f} | {row['PQ']:<10.4f}")
        print("="*80 + "\n")
        
        print("\n" + "="*80)
        print(" RAW DATA DEBUG VIEW (MSMP++ Optimized)")
        print("="*80)
        print(f"{'FOC (frac)':<15} | {'F1':<10} | {'F1*':<10} | {'PC':<10} | {'PQ':<10}")
        print("-" * 65)
        for row in curve_proposed:
            print(f"{row['frac']:<15.6f} | {row['F1']:<10.4f} | {row['F1_star']:<10.4f} | {row['PC']:<10.4f} | {row['PQ']:<10.4f}")
        print("="*80 + "\n")

        # --- RESULTS SUMMARY ---
        curve_baseline = extend_curve(curve_baseline)
        curve_proposed = extend_curve(curve_proposed)

        print("\n=== RESULTS SUMMARY (AUC) ===")
        print(f"{'Metric':<10} | {'MSMP+':<10} | {'MSMP++':<10} | {'Diff'}")
        print("-" * 45)
        for key in ['F1', 'F1_star', 'PQ', 'PC']:
            auc_base = calculate_auc(curve_baseline, x_key='frac', y_key=key)
            auc_imp = calculate_auc(curve_proposed, x_key='frac', y_key=key)
            print(f"{key:<10} | {auc_base:.4f}     | {auc_imp:.4f}     | {auc_imp - auc_base:+.4f}")
    
        plot_results(curve_baseline, curve_proposed)
        plt.show()

    else:
        print("Error: No products loaded from the provided JSON path.")
