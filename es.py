import torch


class RandomEvolutionSolver:
    def __init__(self, named_parameters: dict[str, torch.Tensor], device: torch.device, popsize: int = 100, sigma: float = 0.1, alpha: float = 0.01):
        self.device = device
        self.popsize = popsize
        self.sigma = torch.tensor(sigma).to(device)
        self.alpha = torch.tensor(alpha).to(device)
        self.named_parameters = named_parameters
        for n, p in self.named_parameters.items():
            p.requires_grad = False
        self.parameter_epsilons: dict[str, torch.Tensor] = dict()

    def _set_new_epsilons(self):
        for n, p in self.named_parameters.items():
            self.parameter_epsilons[n] = torch.randn(
                self.popsize, *p.shape, device=self.device)

    def ask(self):
        self._set_new_epsilons()

        solutions: list[dict[str, torch.Tensor]] = []
        for i in range(self.popsize):
            solution = dict()
            for n, p in self.named_parameters.items():
                mutated_p = p + self.sigma * self.parameter_epsilons[n][i]
                solution[n] = mutated_p
            solutions.append(solution)
        return solutions

    def tell(self, rewards: torch.Tensor):
        assert len(rewards) == self.popsize

        # gradient estimation
        g_hat = dict()
        for n, e in self.parameter_epsilons.items():
            g_hat[n] = torch.zeros(e.shape[1:], device=self.device)
            for i in range(len(rewards)):
                # reward_i - reward_mean * epsilon_i (which was used to get that reward)
                g_hat[n] += (rewards[i] - rewards.mean()) * e[i]
            g_hat[n] /= (self.popsize * self.sigma)

        # print(max(tensor.max().item() for tensor in g_hat.values()))

        for n, p in self.named_parameters.items():
            # gradient descent for each parameter
            self.named_parameters[n] -= self.alpha * g_hat[n]
