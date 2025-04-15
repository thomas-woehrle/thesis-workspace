from typing import Optional

import torch
import torch.nn as nn


class OpenAIEvolutionaryOptimizer:
    def __init__(
        self,
        popsize: int,
        sigma: float,
        lr: float,
        model: nn.Module,
        use_antithetic_sampling: bool,
        device: torch.device,
    ):
        self.popsize = popsize
        self.sigma = sigma
        self.lr = lr
        self.use_antithetic_sampling = use_antithetic_sampling
        self.model = model
        self.params_vector = nn.utils.parameters_to_vector(model.parameters())
        self._epsilon = torch.zeros(popsize, len(self.params_vector))
        self.device = device

    def prepare_mutations(self):
        """Creates new epsilon. Epsilon is of shape popsize * num_params"""
        if self.use_antithetic_sampling:
            assert self.popsize % 2 == 0, "If using antithetic sampling, the popsize has to be even"

            # This seemingly weird direct assignment to self._epsilon is done to not waste RAM
            self._epsilon = torch.randn(
                self.popsize // 2, len(self.params_vector), device=self.device
            )
            self._epsilon = torch.concatenate([self._epsilon, -self._epsilon], dim=0)
        else:
            self._epsilon = torch.randn(self.popsize, len(self.params_vector), device=self.device)

    def load_mutation_into_model(self, mutation_idx: int):
        candidate_vector = self.params_vector + self._epsilon[mutation_idx] * self.sigma
        nn.utils.vector_to_parameters(candidate_vector, self.model.parameters())

    def step(self, losses: torch.Tensor):
        # losses of shape popsize x 1
        # estimate gradients
        normalized_losses = (losses - losses.mean()) / losses.std()
        g_hat = (self._epsilon.T @ normalized_losses).flatten()
        g_hat = g_hat / (self.popsize * self.sigma)
        self.params_vector -= self.lr * g_hat
        nn.utils.vector_to_parameters(self.params_vector, self.model.parameters())


class SimpleEvolutionaryOptimizer:
    def __init__(
        self,
        n_parents: int,
        n_children_per_parent: int,
        sigma: float,
        model: nn.Module,
        # use_antithetic_sampling: bool,
        device: torch.device,
    ):
        self.n_parents = n_parents
        self.n_children_per_parent = n_children_per_parent
        self.sigma = sigma
        self.model = model
        self.parent_param_vectors = nn.utils.parameters_to_vector(model.parameters()).repeat(
            n_parents, 1
        )
        self.mutations = torch.zeros(
            n_parents, n_children_per_parent, self.parent_param_vectors.shape[-1]
        )
        self.device = device

    def prepare_mutations(self):
        """Creates new epsilon. Epsilon is of shape popsize * num_params"""
        self.mutations = (
            torch.randn(
                self.n_parents,
                self.n_children_per_parent,
                self.parent_param_vectors.shape[-1],
                device=self.device,
            )
            * self.sigma
        )

    def load_individual_into_model(self, parent_idx: int, mutation_idx: Optional[int]):
        if mutation_idx is None:
            individual_param_vector = self.parent_param_vectors[parent_idx]
        else:
            individual_param_vector = (
                self.parent_param_vectors[parent_idx] + self.mutations[parent_idx][mutation_idx]
            )

        nn.utils.vector_to_parameters(individual_param_vector, self.model.parameters())

    def step(self, losses: torch.Tensor):
        # losses of shape n_parents x (n_children_per_parent + 1)
        # index (p, -1) is assumed to be the loss for the parent itself
        normalized_losses = (losses - losses.mean()) / losses.std()  # keep this for now
        flat_normalized_losses = normalized_losses.flatten()
        # TODO use normal sorting instead of topk?
        _, flat_indices = flat_normalized_losses.topk(self.n_parents, largest=False)
        rows = flat_indices // losses.shape[1]
        cols = flat_indices % losses.shape[1]
        full_indices = torch.stack([rows, cols], dim=1)

        new_parent_param_vectors = torch.zeros_like(self.parent_param_vectors)
        for i, full_idx in enumerate(full_indices):
            # if is parent
            if full_idx[1] == self.n_children_per_parent:
                new_parent_param_vectors[i] = self.parent_param_vectors[full_idx[0]]
            else:
                new_parent_param_vectors[i] = (
                    self.parent_param_vectors[full_idx[0]] + self.mutations[tuple(full_idx)]
                )

        self.parent_param_vectors = new_parent_param_vectors
        nn.utils.vector_to_parameters(self.parent_param_vectors[0], self.model.parameters())
