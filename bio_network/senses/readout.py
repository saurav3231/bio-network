"""Frozen, label-gated readout for unsupervised digit emergence (M3, E3c).

The recurrent weights are learned with STDP *without ever seeing a label*. Once
training is over we attach a readout to translate per-neuron spike responses
into digit classes. This module is deliberately gated so that labels are usable
for one purpose only: assigning a preferred class to each neuron from a
**training-only** response matrix. Evaluating the readout on the held-out test
set never feeds labels back into assignment (the ``LabelsReadout`` object is
frozen at fit time; ``predict``/``predict_vote`` accept only the responses and
the frozen fingerprint).
"""

from __future__ import annotations

import numpy as np


class LabelsReadout:
    """Responsibility-scoped digit readout on top of STDP-tuned responses.

    Two concerns are kept separate:

    - ``fit(train_responses, train_labels)``: compute a per-class mean-response
      fingerprint **and** a hard per-neuron assignment, exclusively from the
      training split (the only place labels are allowed).
    - ``predict(responses)``: the soft prototype classifier -- each class is
      scored by the image's response correlated against that class's mean
      training profile.
    - ``predict_vote(responses)``: the hard "winner per neuron" classifier --
      every neuron fires a vote for its assigned class and the plurality class
      (deterministically) wins. Both are frozen readouts; test labels never
      enter either.
    """

    n_classes: int = 10

    def __init__(self, n_neurons: int, n_classes: int = 10) -> None:
        self.n_neurons = int(n_neurons)
        self.n_classes = int(n_classes)
        self.assignment: np.ndarray | None = None
        self.class_profiles: np.ndarray | None = None

    # -- assignment (labels allowed here only) --------------------------------
    def fit(self, responses: np.ndarray, labels: np.ndarray) -> None:
        """Build the frozen fingerprint from training responses.

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

        means = np.zeros((self.n_classes, self.n_neurons))
        for cls in range(self.n_classes):
            mask = labels == cls
            if mask.any():
                means[cls] = responses[mask].mean(axis=0)
        self.class_profiles = means
        self.assignment = np.argmax(means, axis=0)

    # -- prediction (labels forbidden here) ----------------------------------
    def predict(self, responses: np.ndarray) -> np.ndarray:
        """Soft prototype scoring: image <-> class-profile correlation.

        Each class is scored by the dot product of the image's response vector
        against that class's **frozen** mean training profile.

        Args:
            responses: ``(n_images, n_neurons)`` spike counts for the images to
                classify (test set or frozen training set).

        Returns:
            Per-image predicted class, computed from the frozen fingerprint
            only; no labels are consumed.
        """
        if self.class_profiles is None:
            raise RuntimeError("LabelsReadout.fit() must run before predict()")
        responses = np.asarray(responses, dtype=float)
        scores = responses @ self.class_profiles.T
        return np.argmax(scores, axis=1)

    def predict_vote(self, responses: np.ndarray) -> np.ndarray:
        """Hard plurality vote: each fired neuron votes for its assigned class.

        Every neuron that spiked at least once during a presentation casts one
        vote for ``assignment``; the class with the most votes wins (ties break
        to the lower class index, like ``argmax``). The assignment is frozen at
        ``fit`` time, so votes are a pure function of the response matrix.

        Args:
            responses: ``(n_images, n_neurons)`` spike counts.

        Returns:
            Per-image predicted class from the plurality of frozen votes.
        """
        if self.assignment is None:
            raise RuntimeError("LabelsReadout.fit() must run before predict_vote()")
        responses = np.asarray(responses, dtype=float)
        votes = np.zeros((responses.shape[0], self.n_classes), dtype=np.int64)
        fired = responses > 0
        for i in range(responses.shape[0]):
            rows = np.flatnonzero(fired[i])
            if rows.size:
                votes[i] = np.bincount(self.assignment[rows], minlength=self.n_classes)
        return np.argmax(votes, axis=1)
