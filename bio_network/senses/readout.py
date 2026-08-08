"""Frozen, label-gated readout for unsupervised digit emergence (M3, E3c).

The recurrent weights are learned with STDP *without ever seeing a label*. Once
training is over we attach a readout to translate per-neuron spike responses
into digit classes. This module is deliberately gated so that labels are usable
for one purpose only: assigning a preferred class to each neuron from a
**training-only** response matrix. Evaluating the readout on the held-out test
set never feeds labels back into assignment (the ``LabelsReadout`` object is
frozen at fit time; ``score_test`` accepts only the responses and the frozen
assignment).
"""

from __future__ import annotations

import numpy as np


class LabelsReadout:
    """Responsibility-scoped digit readout on top of STDP-tuned responses.

    Wraps two separate concerns:

    - ``fit(train_responses, train_labels)``: assign each neuron its
      highest-mean-response class, computed exclusively from the training
      split (this is the single place labels are allowed).
    - ``predict(responses)``: classify new (test) observations using only the
      frozen assignment, never touching the test labels.
    """

    n_classes: int = 10

    def __init__(self, n_neurons: int, n_classes: int = 10) -> None:
        self.n_neurons = int(n_neurons)
        self.n_classes = int(n_classes)
        self.assignment: np.ndarray | None = None
        self.class_profiles: np.ndarray | None = None

    # -- assignment (labels allowed here only) --------------------------------
    def fit(self, responses: np.ndarray, labels: np.ndarray) -> None:
        """Assign each neuron its best class from training responses.

        Args:
            responses: ``(n_images, n_neurons)`` per-neuron spike counts on a
                training set.
            labels: ``(n_images,)`` integer class labels of that same set.
        """
        responses = np.asarray(responses, dtype=float)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if responses.shape[1] != self.n_neurons:
            raise ValueError(
                f"response width {responses.shape[1]} != n_neurons {self.n_neurons}"
            )

        # Mean response per (class, neuron).
        means = np.zeros((self.n_classes, self.n_neurons))
        for cls in range(self.n_classes):
            mask = labels == cls
            if mask.any():
                means[cls] = responses[mask].mean(axis=0)
        self.class_profiles = means

        # A neuron's class = the class that, on average, drives it hardest.
        self.assignment = np.argmax(means, axis=0)

    # -- prediction (labels forbidden here) ----------------------------------
    def predict(self, responses: np.ndarray) -> np.ndarray:
        """Classify held-out responses using only the frozen profile.

        Each class is scored by the dot product of the image's response vector
        against that class's mean training response profile (its neuron-wise
        activity fingerprint). This is the statistically better soft readout
        compared to a hard win-per-neuron assignment, and it remains frozen:
        the profiles were built exclusively from the training split.
        """
        if self.class_profiles is None:
            raise RuntimeError("LabelsReadout.fit() must run before predict()")
        responses = np.asarray(responses, dtype=float)
        scores = responses @ self.class_profiles.T
        return np.argmax(scores, axis=1)
