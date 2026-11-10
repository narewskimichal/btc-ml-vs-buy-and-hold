import json
from pathlib import Path

import numpy as np

SMALL_QUICKSORT = 15


def load_notebook_code(path, marker=None):
    """Zwraca kod z komórek notatnika (opcjonalnie tylko z komórek zawierających marker)."""
    nb = json.loads(Path(path).read_text(encoding='utf-8'))
    cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    if marker is not None:
        cells = [c for c in cells if marker in c]
    return '\n'.join(cells)


def _less(a, b):
    return a < b or (b != b and a == a)


def _heap_argsort(v, idx, lo, hi):
    n = hi - lo + 1
    a = idx[lo:hi + 1]

    def sift(start, end):
        tmp = a[start]
        i, j = start, 2 * start + 1
        while j <= end:
            if j < end and _less(v[a[j]], v[a[j + 1]]):
                j += 1
            if _less(v[tmp], v[a[j]]):
                a[i] = a[j]
                i, j = j, 2 * j + 1
            else:
                break
        a[i] = tmp

    for start in range(n // 2 - 1, -1, -1):
        sift(start, n - 1)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift(0, end - 1)
    idx[lo:hi + 1] = a


def baseline_argsort(values):
    """Argsort zgodny z klasycznym quicksortem numpy (bez przyspieszeń AVX).

    W procedurze adaptacyjnej wiele konfiguracji ma identyczny wynik na historii. Kolejność
    takich remisów zależy od wersji numpy i procesora, a wyniki pracy policzono przy
    klasycznym algorytmie. Ta funkcja odtwarza go niezależnie od komputera.
    """
    v = np.asarray(values, dtype=float).tolist()
    idx = list(range(len(v)))
    if len(v) < 2:
        return np.array(idx, dtype=np.intp)

    stack = []
    lo, hi = 0, len(v) - 1
    depth = (len(v).bit_length() - 1) * 2
    while True:
        if depth < 0:
            _heap_argsort(v, idx, lo, hi)
        else:
            while hi - lo > SMALL_QUICKSORT:
                mid = lo + ((hi - lo) >> 1)
                if _less(v[idx[mid]], v[idx[lo]]):
                    idx[mid], idx[lo] = idx[lo], idx[mid]
                if _less(v[idx[hi]], v[idx[mid]]):
                    idx[hi], idx[mid] = idx[mid], idx[hi]
                if _less(v[idx[mid]], v[idx[lo]]):
                    idx[mid], idx[lo] = idx[lo], idx[mid]
                pivot = v[idx[mid]]
                i, j = lo, hi - 1
                idx[mid], idx[j] = idx[j], idx[mid]
                while True:
                    i += 1
                    while _less(v[idx[i]], pivot):
                        i += 1
                    j -= 1
                    while _less(pivot, v[idx[j]]):
                        j -= 1
                    if i >= j:
                        break
                    idx[i], idx[j] = idx[j], idx[i]
                k = hi - 1
                idx[i], idx[k] = idx[k], idx[i]
                depth -= 1
                if i - lo < hi - i:
                    stack.append((i + 1, hi, depth))
                    hi = i - 1
                else:
                    stack.append((lo, i - 1, depth))
                    lo = i + 1
            for i in range(lo + 1, hi + 1):
                item = idx[i]
                key = v[item]
                j = i
                while j > lo and _less(key, v[idx[j - 1]]):
                    idx[j] = idx[j - 1]
                    j -= 1
                idx[j] = item
        if not stack:
            break
        lo, hi, depth = stack.pop()
    return np.array(idx, dtype=np.intp)


def load_adaptive_procedure(notebook_path, namespace):
    """Wczytuje procedurę adaptacyjną z notatnika pracy, z sortowaniem zgodnym z wynikami pracy."""
    source = load_notebook_code(notebook_path, marker='def procedura_glowna')
    source = source.replace('np.argsort(', 'baseline_argsort(')
    namespace['baseline_argsort'] = baseline_argsort
    exec(source, namespace)
