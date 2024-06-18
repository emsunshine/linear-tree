import torch


SCORING = {
    'linear': lambda y, yh: y - yh,
    'square': lambda y, yh: torch.square(y - yh),
    'absolute': lambda y, yh: torch.abs(y - yh),
    'square_logarithmic': lambda y, yh: torch.square(torch.log10(y.clip(1e-6) + 1) - torch.log10(yh.clip(1e-6) + 1)),
    'exponential': lambda y, yh: 1 - torch.exp(-torch.abs(y - yh)),
    'poisson': lambda y, yh: yh.clip(1e-6) - y * torch.log(yh.clip(1e-6)),
    'hamming': lambda y, yh, classes: (y != yh).astype(int),
    'entropy': lambda y, yh, classes: torch.sum(list(map(
        lambda c: -(y == c[1]).astype(int) * torch.log(yh[:, c[0]]),
        enumerate(classes))), axis=0)
}


def _normalize_score(scores, weights=None):
    """Normalize scores according to weights"""

    if weights is None:
        return scores.mean()
    else:
        return torch.mean(torch.dot(scores.T, weights) / weights.sum())


def mse(model, X, y, weights=None, **largs):
    """Mean Squared Error"""

    pred = model.predict(X)
    scores = SCORING['square'](y, pred)

    return _normalize_score(scores, weights)


def rmse(model, X, y, weights=None, **largs):
    """Root Mean Squared Error"""

    return torch.sqrt(mse(model, X, y, weights, **largs))


def mae(model, X, y, weights=None, **largs):
    """Mean Absolute Error"""

    pred = model.predict(X)
    scores = SCORING['absolute'](y, pred)

    return _normalize_score(scores, weights)

def msle(model, X, y, weights=None, **largs):
    """Root Mean Squared Logarithmic Error"""

    pred = model.predict(X)
    scores = SCORING['square_logarithmic'](y, pred)

    return _normalize_score(scores, weights)


def poisson(model, X, y, weights=None, **largs):
    """Poisson Loss"""

    if torch.any(y < 0):
        raise ValueError("Some value(s) of y are negative which is"
                         " not allowed for Poisson regression.")

    pred = model.predict(X)
    scores = SCORING['poisson'](y, pred)

    return _normalize_score(scores, weights)


def hamming(model, X, y, weights=None, **largs):
    """Hamming Loss"""

    pred = model.predict(X)
    scores = SCORING['hamming'](y, pred, None)

    return _normalize_score(scores, weights)


def crossentropy(model, X, y, classes, weights=None, **largs):
    """Cross Entropy Loss"""

    pred = model.predict_proba(X).clip(1e-5, 1 - 1e-5)
    scores = SCORING['entropy'](y, pred, classes)

    return _normalize_score(scores, weights)