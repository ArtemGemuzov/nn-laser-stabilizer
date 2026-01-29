from nn_laser_stabilizer.policy.policy import Policy
from nn_laser_stabilizer.model.actor import Actor
from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import ExplorationType
from nn_laser_stabilizer.policy.deterministic import DeterministicPolicy
from nn_laser_stabilizer.policy.exploration import NoisyExplorationPolicy, OrnsteinUhlenbeckExplorationPolicy, RandomExplorationPolicy


def make_policy_from_config(
    actor: Actor,
    exploration_config: Config,
) -> Policy:
    exploration_type = ExplorationType.from_str(exploration_config.type)
    exploration_steps = exploration_config.steps
    
    if exploration_type == ExplorationType.NONE:
        if exploration_steps != 0:
            raise ValueError(
                f"exploration_steps must be 0 when exploration_type is {ExplorationType.NONE}, "
                f"got exploration_steps={exploration_steps}"
            )
        return DeterministicPolicy(actor=actor)
    elif exploration_type == ExplorationType.RANDOM:
        return RandomExplorationPolicy(
            actor=actor,
            exploration_steps=exploration_steps,
        )
    elif exploration_type == ExplorationType.NOISY:
        policy_noise = exploration_config.policy_noise
        noise_clip = exploration_config.noise_clip
        
        if policy_noise <= 0.0:
            raise ValueError(
                f"policy_noise must be greater than 0 when exploration_type is {ExplorationType.NOISY}, "
                f"got policy_noise={policy_noise}"
            )
        if noise_clip <= 0.0:
            raise ValueError(
                f"noise_clip must be greater than 0 when exploration_type is {ExplorationType.NOISY}, "
                f"got noise_clip={noise_clip}"
            )
        
        return NoisyExplorationPolicy(
            actor=actor,
            exploration_steps=exploration_steps,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
        )
    elif exploration_type == ExplorationType.OU:
        sigma = exploration_config.sigma
        theta = exploration_config.theta
        mu = exploration_config.mu
        dt = exploration_config.dt

        if sigma <= 0.0:
            raise ValueError(
                f"sigma must be greater than 0 when exploration_type is {ExplorationType.OU}, "
                f"got sigma={sigma}"
            )

        if theta <= 0.0:
            raise ValueError(
                f"theta must be greater than 0 when exploration_type is {ExplorationType.OU}, "
                f"got theta={theta}"
            )

        if dt <= 0.0:
            raise ValueError(
                f"dt must be greater than 0 when exploration_type is {ExplorationType.OU}, "
                f"got dt={dt}"
            )

        return OrnsteinUhlenbeckExplorationPolicy(
            actor=actor,
            exploration_steps=exploration_steps,
            theta=theta,
            sigma=sigma,
            mu=mu,
            dt=dt,
        )
    else:
        raise ValueError(f"Unknown exploration type: {exploration_type}")