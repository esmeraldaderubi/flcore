
from typing import List, Tuple
 
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from tqdm import tqdm
 
from flcore.metrics import calculate_metrics
 
NDArrays = List[np.ndarray]
 
 
def _bce(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    """Binary cross-entropy on probabilities (numpy, torch-free)."""
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
 
 
class MLPModel:
    """scikit-learn MLP wrapper exposing a Flower-friendly weight interface.
 
    Drop-in replacement for the old torch `CNN`: same constructor signature and
    the same `get_weights` / `set_weights` contract (a flat list of NumPy arrays).
    """
 
    def __init__(
        self,
        client_num: int = 5,
        client_tree_num: int = 100,
        n_channel: int = 64,
        task_type: str = "BINARY",
    ) -> None:
        self.task_type = task_type
        self.input_dim = client_num * client_tree_num
        # The old conv produced n_channel features per client block, flattened
        # across all clients -> n_channel * client_num hidden units.
        hidden = (n_channel * client_num,)
        self.n_layers = len(hidden) + 1  # coef/intercept arrays per side
 
        common = dict(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            learning_rate_init=1e-4,  # matches the old Adam lr
            alpha=0.0,                # no L2, the torch net had none
            random_state=0,           # identical init across clients for FedAvg
            max_iter=1,
        )
 
        if task_type == "BINARY":
            self.estimator = MLPClassifier(**common)
        elif task_type == "REG":
            self.estimator = MLPRegressor(**common)
        else:
            # The original CNN had n_out = 1, so it only supported BINARY/REG.
            raise ValueError(f"Unsupported task_type for this model: {task_type}")
 
        self._initialize()
 
    def _initialize(self) -> None:
        """Force weight allocation so get_weights works before any real fit.
 
        Uses a deterministic dummy partial_fit; because random_state is fixed and
        the dummy data is identical, every client starts from the same weights.
        """
        x0 = np.zeros((2, self.input_dim), dtype=np.float32)
        if self.task_type == "BINARY":
            self.estimator.partial_fit(x0, np.array([0, 1]), classes=np.array([0, 1]))
        else:
            self.estimator.partial_fit(x0, np.array([0.0, 1.0]))
 
    def get_weights(self) -> NDArrays:
        """Coefs first, then intercepts — a flat list of NumPy arrays."""
        coefs = [np.array(w, copy=True) for w in self.estimator.coefs_]
        intercepts = [np.array(b, copy=True) for b in self.estimator.intercepts_]
        return coefs + intercepts
 
    def set_weights(self, weights: NDArrays) -> None:
        """Inverse of get_weights. Same split point on every client."""
        coefs = weights[: self.n_layers]
        intercepts = weights[self.n_layers : 2 * self.n_layers]
        self.estimator.coefs_ = [np.array(w, copy=True) for w in coefs]
        self.estimator.intercepts_ = [np.array(b, copy=True) for b in intercepts]
        # Drop the stale Adam state so the next training round rebinds to the
        # freshly assigned weight arrays (the torch version also built a new
        # optimizer every fit() call).
        if hasattr(self.estimator, "_optimizer"):
            del self.estimator._optimizer
 
 
# Backwards-compatible alias for any code still importing `CNN`.
CNN = MLPModel
 
 
def train(
    task_type: str,
    net: MLPModel,
    trainset,  # TreeDataset-like: has .data and .labels
    num_iterations: int,
    log_progress: bool = True,
) -> Tuple[float, float, int]:
    """Full-batch training for `num_iterations` Adam steps."""
    x = np.asarray(trainset.data, dtype=np.float32)
    y = np.asarray(trainset.labels)
    n_samples = x.shape[0]
    num_iterations = int(num_iterations) if num_iterations else 1
 
    # Fresh optimizer for this round, then full-batch updates.
    if hasattr(net.estimator, "_optimizer"):
        del net.estimator._optimizer
    net.estimator.batch_size = n_samples
 
    iterator = range(num_iterations)
    if log_progress:
        iterator = tqdm(iterator, total=num_iterations, desc="TRAIN")
 
    if task_type == "BINARY":
        y = y.astype(int).ravel()
        for _ in iterator:
            net.estimator.partial_fit(x, y)
        proba = net.estimator.predict_proba(x)[:, 1]
        loss = _bce(y, proba)
        preds = (proba >= 0.5).astype(int)
        result = float(accuracy_score(y, preds))
    elif task_type == "REG":
        y = y.astype(np.float32).ravel()
        for _ in iterator:
            net.estimator.partial_fit(x, y)
        preds = net.estimator.predict(x)
        loss = float(mean_squared_error(y, preds))
        result = loss
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
 
    if log_progress:
        print("\n")
 
    return loss, result, n_samples
 
 
def test(
    task_type: str,
    net: MLPModel,
    testset,  # TreeDataset-like: has .data and .labels
    log_progress: bool = True,
) -> Tuple[float, dict, int]:
    """Evaluate on test/val data. Metrics come from flcore.metrics."""
    x = np.asarray(testset.data, dtype=np.float32)
    y = np.asarray(testset.labels)
    n_samples = x.shape[0]
 
    if task_type == "BINARY":
        y_int = y.astype(int).ravel()
        proba = net.estimator.predict_proba(x)[:, 1]
        loss = _bce(y_int, proba)
        preds = (proba >= 0.5).astype(int)
        metrics = calculate_metrics(y_int, preds, task_type)
    elif task_type == "REG":
        y_f = y.astype(np.float32).ravel()
        preds = net.estimator.predict(x)
        loss = float(mean_squared_error(y_f, preds))
        metrics = calculate_metrics(y_f, preds, task_type)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
 
    metrics = {k: float(v) for k, v in metrics.items()}
 
    if log_progress:
        print("\n")
 
    return loss, metrics, n_samples
 
 
def print_model_layers(model: MLPModel) -> None:
    est = model.estimator
    print(est)
    for i, w in enumerate(getattr(est, "coefs_", [])):
        print(f"coef_{i}\t{w.shape}")
    for i, b in enumerate(getattr(est, "intercepts_", [])):
        print(f"intercept_{i}\t{b.shape}")
 