import logging

import torch
from torch import Tensor, nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, TensorDataset

from tmrl_openpi.models_pytorch.pi0_pytorch import PI0Pytorch

logger = logging.getLogger("tmrl_openpi")


class EnsembleMember(nn.Module):
    """MLP that predicts an action sequence from a low-dim state vector."""

    def __init__(self, obs_dim: int, action_dim: int, action_horizon: int, hidden_size: int = 512, num_layers: int = 3):
        super().__init__()
        out_dim = action_horizon * action_dim
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs).view(-1, self.action_horizon, self.action_dim)


class PosteriorVarianceEstimator(nn.Module):
    """Bootstrapped ensemble that estimates per-state posterior variance over actions.

    Workflow:
        1. Call ``fit(states, actions)`` on your offline dataset.
        2. During policy training call ``compute_std(states)`` to get
           per-dimension standard deviations ``[B, action_horizon, action_dim]``.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_horizon: int,
        ensemble_size: int = 10,
        hidden_size: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.members = nn.ModuleList(
            [EnsembleMember(obs_dim, action_dim, action_horizon, hidden_size, num_layers) for _ in range(ensemble_size)]
        )
        self._fitted = False

    def fit(
        self,
        states: Tensor,
        actions: Tensor,
        num_epochs: int = 1000,
        lr: float = 3e-4,
        batch_size: int = 256,
        device: str | torch.device = "cuda",
    ) -> None:
        """Train each member on a bootstrapped resample of ``(states, actions)``.

        Args:
            states:  ``[N, obs_dim]``
            actions: ``[N, action_horizon, action_dim]``
        """
        device = torch.device(device)
        self.to(device)
        n = states.shape[0]

        for k, member in enumerate(self.members):
            idx = torch.randint(0, n, (n,))
            b_states = states[idx].to(device)
            b_actions = actions[idx].to(device)

            optimizer = torch.optim.Adam(member.parameters(), lr=lr)
            loader = DataLoader(TensorDataset(b_states, b_actions), batch_size=batch_size, shuffle=True)

            member.train()
            for _epoch in range(num_epochs):
                for obs_batch, act_batch in loader:
                    loss = F.mse_loss(member(obs_batch), act_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            logger.info("Ensemble member %d/%d trained.", k + 1, self.ensemble_size)

        for param in self.parameters():
            param.requires_grad = False
        self._fitted = True

    @torch.no_grad()
    def compute_std(self, states: Tensor) -> Tensor:
        """Return per-dimension posterior std: ``[B, action_horizon, action_dim]``."""
        preds = torch.stack([m(states) for m in self.members], dim=0)  # [K, B, H, D]
        var = preds.var(dim=0)  # [B, H, D]
        return var.sqrt()


class PostBCPytorch(PI0Pytorch):
    """PI0 flow-matching policy with POSTBC action relabeling.

    Identical to ``PI0Pytorch`` except that, when the posterior variance
    ensemble has been fitted, every training forward pass perturbs the
    ground-truth action targets by ``alpha * N(0, posterior_std(s))``
    before computing the flow-matching loss.  At inference time
    (``sample_actions``) nothing changes.
    """

    def __init__(
        self,
        config,
        *,
        ensemble_size: int = 10,
        ensemble_hidden: int = 512,
        ensemble_layers: int = 3,
        alpha: float = 1.0,
    ):
        super().__init__(config)
        self.alpha = alpha
        self.variance_estimator = PosteriorVarianceEstimator(
            obs_dim=config.action_dim,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            ensemble_size=ensemble_size,
            hidden_size=ensemble_hidden,
            num_layers=ensemble_layers,
        )

    def fit_ensemble(self, states: Tensor, actions: Tensor, **kwargs) -> None:
        """Pre-train the variance ensemble.  Call once before main policy training.

        Args:
            states:  ``[N, state_dim]``  (typically ``observation.state``)
            actions: ``[N, action_horizon, action_dim]``
        """
        self.variance_estimator.fit(states, actions, **kwargs)

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        if self.variance_estimator._fitted and self.training:
            with torch.no_grad():
                std = self.variance_estimator.compute_std(observation.state.float())
                perturbation = torch.randn_like(actions) * std
            actions = actions + self.alpha * perturbation

        return super().forward(observation, actions, noise=noise, time=time)
