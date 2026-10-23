"""json.dump qui ne produit jamais de JSON invalide : NaN / +-Inf -> null (les navigateurs refusent NaN)."""
import json, math


def clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def dump(obj, fp, **kw):
    kw.setdefault("allow_nan", False)
    return json.dump(clean(obj), fp, **kw)
