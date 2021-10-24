import gym
import numpy as np
import torch
from utils import *
import torch
import torch.nn as nn

def evaluate(policy, num_episodes=1, seed=0, env_name='FrozenLake8x8-v1',
             render=False, existing_env=None):
    """This function evaluate the given policy and return the mean episode
    reward.
    :param policy: a function whose input is the observation
    :param num_episodes: number of episodes you wish to run
    :param seed: the random seed
    :param env_name: the name of the environment
    :param render: a boolean flag indicating whether to render policy
    :return: the averaged episode reward of the given policy.
    """
    if existing_env is None:
        env = gym.make(env_name)
        env.seed(seed)
    else:
        env = existing_env
    rewards = []
    if render: num_episodes = 1
    for i in range(num_episodes):
        obs = env.reset()
        act = policy(obs)
        ep_reward = 0
        while True:
            obs, reward, done, info = env.step(act)
            act = policy(obs)
            ep_reward += reward
            if render:
                env.render()
                wait(sleep=0.05)
            if done:
                break
        rewards.append(ep_reward)
    if render:
        env.close()
    return np.mean(rewards)

def run(trainer_cls, config=None, reward_threshold=None):
    """Run the trainer and report progress, agnostic to the class of trainer
    :param trainer_cls: A trainer class
    :param config: A dict
    :param reward_threshold: the reward threshold to break the training
    :return: The trained trainer and a dataframe containing learning progress
    """
    if config is None:
        config = {}
    trainer = trainer_cls(config)
    config = trainer.config
    start = now = time.time()
    stats = []
    for i in range(config['max_iteration'] + 1):
        stat = trainer.train()
        stats.append(stat or {})
        if i % config['evaluate_interval'] == 0 or \
                i == config["max_iteration"]:
            reward = trainer.evaluate(config.get("evaluate_num_episodes", 50))
            print("({:.1f}s,+{:.1f}s)\tIteration {}, current mean episode "
                  "reward is {}. {}".format(
                time.time() - start, time.time() - now, i, reward,
                {k: round(np.mean(v), 4) for k, v in
                 stat.items()} if stat else ""))
            now = time.time()
        if reward_threshold is not None and reward > reward_threshold:
            print("In {} iteration, current mean episode reward {:.3f} is "
                  "greater than reward threshold {}. Congratulation! Now we "
                  "exit the training process.".format(
                i, reward, reward_threshold))
            break
    return trainer, stats

default_config = dict(
    env_name="CartPole-v0",
    max_iteration=1000,
    max_episode_length=1000,
    evaluate_interval=100,
    gamma=0.99,
    eps=0.3,
    seed=0
)


class AbstractTrainer:
    """This is the abstract class for value-based RL trainer. We will inherent
    the specify algorithm's trainer from this abstract class, so that we can
    reuse the codes.
    """

    def __init__(self, config):
        self.config = merge_config(config, default_config)

        # Create the environment
        self.env_name = self.config['env_name']
        self.env = gym.make(self.env_name)
        if self.env_name == "Pong-ram-v0":
            self.env = wrap_deepmind_ram(self.env)

        # Apply the random seed
        self.seed = self.config["seed"]
        np.random.seed(self.seed)
        self.env.seed(self.seed)

        # We set self.obs_dim to the number of possible observation
        # if observation space is discrete, otherwise the number
        # of observation's dimensions. The same to self.act_dim.
        if isinstance(self.env.observation_space, gym.spaces.box.Box):
            assert len(self.env.observation_space.shape) == 1
            self.obs_dim = self.env.observation_space.shape[0]
            self.discrete_obs = False
        elif isinstance(self.env.observation_space,
                        gym.spaces.discrete.Discrete):
            self.obs_dim = self.env.observation_space.n
            self.discrete_obs = True
        else:
            raise ValueError("Wrong observation space!")

        if isinstance(self.env.action_space, gym.spaces.box.Box):
            assert len(self.env.action_space.shape) == 1
            self.act_dim = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.act_dim = self.env.action_space.n
        else:
            raise ValueError("Wrong action space!")

        self.eps = self.config['eps']

        # You need to setup the parameter for your function approximator.
        self.initialize_parameters()

    def initialize_parameters(self):
        self.parameters = None
        raise NotImplementedError(
            "You need to override the "
            "Trainer._initialize_parameters() function.")

    def process_state(self, state):
        """Preprocess the state (observation).

        If the environment provides discrete observation (state), transform
        it to one-hot form. For example, the environment FrozenLake-v0
        provides an integer in [0, ..., 15] denotes the 16 possible states.
        We transform it to one-hot style:

        original state 0 -> one-hot vector [1, 0, 0, 0, 0, 0, 0, 0, ...]
        original state 1 -> one-hot vector [0, 1, 0, 0, 0, 0, 0, 0, ...]
        original state 15 -> one-hot vector [0, ..., 0, 0, 0, 0, 0, 1]

        If the observation space is continuous, then you should do nothing.
        """
        if not self.discrete_obs:
            return state
        else:
            new_state = np.zeros((self.obs_dim,))
            new_state[state] = 1
        return new_state

    def compute_values(self, processed_state):
        """Approximate the state value of given state.
        This is a private function.
        Note that you should NOT preprocess the state here.
        """
        raise NotImplementedError("You need to override the "
                                  "Trainer.compute_values() function.")

    def compute_action(self, processed_state, eps=None):
        """Compute the action given the state. Note that the input
        is the processed state."""

        values = self.compute_values(processed_state)
        assert values.ndim == 1, values.shape

        if eps is None:
            eps = self.eps

        # [TODO] Implement the epsilon-greedy policy here. We have `eps`
        #  probability to choose a uniformly random action in action_space,
        #  otherwise choose action that maximizes the values.
        # Hint: Use the function of self.env.action_space to sample random
        # action.

        policy_table = self.eps * np.ones(self.env.action_space.n) / self.env.action_space.n
        idx_max = np.argmax(values)
        policy_table[idx_max] += 1 - self.eps
        action = np.random.choice(np.arange(self.env.action_space.n), p=policy_table)

        return action

    def evaluate(self, num_episodes=50, *args, **kwargs):
        """Use the function you write to evaluate current policy.
        Return the mean episode reward of 50 episodes."""
        policy = lambda raw_state: self.compute_action(
            self.process_state(raw_state), eps=0.0)
        if "MetaDrive" in self.env_name:
            kwargs["existing_env"] = self.env
        result = evaluate(policy, num_episodes, seed=self.seed,
                          env_name=self.env_name, *args, **kwargs)
        return result

    def compute_gradient(self, *args, **kwargs):
        """Compute the gradient."""
        raise NotImplementedError(
            "You need to override the Trainer.compute_gradient() function.")

    def apply_gradient(self, *args, **kwargs):
        """Compute the gradient"""
        raise NotImplementedError(
            "You need to override the Trainer.apply_gradient() function.")

    def train(self):
        """Conduct one iteration of learning."""
        raise NotImplementedError("You need to override the "
                                  "Trainer.train() function.")

linear_approximator_config = merge_config(dict(
    parameter_std=0.01,
    learning_rate=0.01,
    n=3,
), default_config)


# Solve the TODOs and remove `pass`

# Build the algorithm-specify config.
linear_approximator_config = merge_config(dict(
    parameter_std=0.01,
    learning_rate=0.01,
    n=3,
), default_config)

########### linear ############
class LinearTrainer(AbstractTrainer):
    def __init__(self, config):
        config = merge_config(config, linear_approximator_config)

        # Initialize the abstract class.
        super().__init__(config)

        self.max_episode_length = self.config["max_episode_length"]
        self.learning_rate = self.config["learning_rate"]
        self.gamma = self.config["gamma"]
        self.n = self.config["n"]

    def initialize_parameters(self):
        # [TODO] Initialize self.parameters, which is two dimensional matrix,
        #  and subjects to a normal distribution with scale
        #  config["parameter_std"].
        std = self.config["parameter_std"]
        self.parameters = np.random.normal(scale=std, size=(self.obs_dim, self.act_dim))

        print("Initialize parameters with shape: {}.".format(
            self.parameters.shape))

    def compute_values(self, processed_state):
        # [TODO] Compute the value for each potential action. Note that you
        #  should NOT preprocess the state here."""
        assert processed_state.ndim == 1, processed_state.shape

        ret = np.matmul(self.parameters.T, processed_state)

        return ret

    def train(self):
        """
        Please implement the n-step Sarsa algorithm presented in Chapter 10.2
        of the textbook. You algorithm should reduce the convention one-step
        Sarsa when n = 1. That is:
            TD = r_t + gamma * Q(s_t+1, a_t+1) - Q(s_t, a_t)
            Q(s_t, a_t) = Q(s_t, a_t) + learning_rate * TD
        """
        s = self.env.reset()
        processed_s = self.process_state(s)
        processed_states = [processed_s]
        rewards = [0.0]
        actions = [self.compute_action(processed_s)]
        T = float("inf")

        for t in range(self.max_episode_length):
            if t < T:
                # [TODO]  When the termination is not reach, apply action,
                #  process state, record state / reward / action to the
                #  lists defined above, and deal with termination.
                next_state, reward, done, _ = self.env.step(actions[-1])

                processed_s = self.process_state(next_state)
                processed_states.append(processed_s)
                rewards.append(reward)
                if done:
                    break
                else:
                    next_act = self.compute_action(processed_s)
                    actions.append(next_act)

            tau = t - self.n + 1
            if tau >= 0:
                gradient = self.compute_gradient(
                    processed_states, actions, rewards, tau, T
                )
                self.apply_gradient(gradient)
            if tau == T - 1:
                break

    def compute_gradient(self, processed_states, actions, rewards, tau, T):
        """Compute the gradient"""
        n = self.n

        # [TODO] Compute the approximation goal, the truth state action value
        #  G. It is a n-step discounted sum of rewards. Refer to Chapter 10.2
        #  of the textbook.
        # [HINT] G have two parts: the accumuted reward computed from step tau to
        #  step tau+n, and the possible state value at time step tau+n, if the episode
        #  is not terminated. Remember to apply the discounter factor (\gamma^n) to
        #  the second part of G if applicable.
        valid_states = processed_states[-n-1:]
        valid_rewards = rewards[-n:]

        q_target = sum([reward * self.gamma ** i for i, reward in enumerate(valid_rewards)])
        G = np.ones((self.act_dim, 1)) * q_target

        if tau + n < T:
            # [TODO] If at time step tau + n the episode is not terminated,
            # then we should add the state action value at tau + n
            # to the G.
            next_n_values = self.compute_values(valid_states[-1])
            G += self.gamma ** n * next_n_values[:,None]

        # Denote the state-action value function Q, then the loss of
        # prediction error w.r.t. the weights can be separated into two
        # parts (the chain rule):
        #     dLoss / dweight = (dLoss / dQ) * (dQ / dweight)
        # We call the first one loss_grad, and the latter one
        # value_grad. We consider the Mean Square Error between the target
        # value (G) and the predicted value (Q(s_t, a_t)) to be the loss.

        loss_grad = np.zeros((self.act_dim, 1))
        # [TODO] fill the propoer value of loss_grad, denoting the gradient
        # of the MSE w.r.t. the output of the linear function.
        current_state = valid_states[0]
        current_q = self.compute_values(current_state)[:, None]
        loss_grad = (G - current_q)

        # [TODO] compute the value of value_grad, denoting the gradient of
        # the output of the linear function w.r.t. the parameters.
        value_grad = current_state[:,None]

        assert loss_grad.shape == (self.act_dim, 1)
        assert value_grad.shape == (self.obs_dim, 1)

        # [TODO] merge two gradients to get the gradient of loss w.r.t. the
        # parameters.
        gradient = np.matmul(value_grad, loss_grad.T)
        return gradient

    def apply_gradient(self, gradient):
        """Apply the gradient to the parameter."""
        assert gradient.shape == self.parameters.shape, (
            gradient.shape, self.parameters.shape)
        # [TODO] apply the gradient to self.parameters
        self.parameters += self.learning_rate * gradient

########### polynomial_feature #############

linear_fc_config = merge_config(dict(
    polynomial_order=1,
), linear_approximator_config)


def polynomial_feature(sequence, order=1):
    """
    Construct the order-n polynomial-basis feature of the state.
    Refer to Chapter 9.5.1 of the textbook.
    We expect to get a vector of length `(order+1)^k` as the output,
    wherein `k` is the dimensions of the state.

    For example:
    When the state is [2, 3, 4] (so k=3),
    the first order polynomial feature of the state is
    [
        1,
        2,
        3,
        4,
        2 * 3 = 6,
        2 * 4 = 8,
        3 * 4 = 12,
        2 * 3 * 4 = 24
    ].

    We have `(1+1)^3=8` output dimensions.

    Note: it is not necessary to follow the ascending order.
    """
    # [TODO] finish this function.
    output = []
    k = len(sequence)
    if k > 1:
        v = sequence[0]
        seq = [v ** i for i in range(order + 1)]
        remain_seqs = polynomial_feature(sequence[1:], order=order)
        for s in seq:
            output += [s * r for r in remain_seqs]
    elif k == 1:
        v = sequence[0]
        return [v ** i for i in range(order + 1)]

    return output


assert sorted(polynomial_feature([2, 3, 4])) == [1, 2, 3, 4, 6, 8, 12, 24]
assert len(polynomial_feature([2, 3, 4], 2)) == 27
assert len(polynomial_feature([2, 3, 4], 3)) == 64


class LinearTrainerWithFeatureConstruction(LinearTrainer):
    """In this class, we will expand the dimension of the state.
    This procedure is done at self.process_state function.
    The modification of self.obs_dim and the shape of parameters
    is also needed.
    """

    def __init__(self, config):
        config = merge_config(config, linear_fc_config)
        # Initialize the abstract class.
        super().__init__(config)

        self.polynomial_order = self.config["polynomial_order"]

        # Expand the size of observation
        self.obs_dim = (self.polynomial_order + 1) ** self.obs_dim

        # Since we change self.obs_dim, reset the parameters.
        self.initialize_parameters()

    def process_state(self, state):
        """Please finish the polynomial function."""
        processed = polynomial_feature(state, self.polynomial_order)
        processed = np.asarray(processed)
        assert len(processed) == self.obs_dim, processed.shape
        return processed


######## MLP ###########
mlp_trainer_config = merge_config(dict(
    parameter_std=0.01,
    learning_rate=0.01,
    hidden_dim=100,
    n=3,
    clip_norm=1.0,
    clip_gradient=True
), default_config)


class MLPTrainer(LinearTrainer):
    def __init__(self, config):
        config = merge_config(config, mlp_trainer_config)
        self.hidden_dim = config["hidden_dim"]
        super().__init__(config)

    def initialize_parameters(self):
        # [TODO] Initialize self.hidden_parameters and self.output_parameters,
        #  which are two dimensional matrices, and subject to normal
        #  distributions with scale config["parameter_std"]
        std = self.config["parameter_std"]
        self.hidden_parameters = np.random.normal(scale=std, size=(self.obs_dim, self.hidden_dim))
        self.output_parameters = np.random.normal(scale=std, size=(self.hidden_dim, self.act_dim))

    def compute_values(self, processed_state):
        """[TODO] Compute the value for each potential action. Note that you
        should NOT preprocess the state here."""
        assert processed_state.ndim == 1, processed_state.shape
        activation = self.compute_activation(processed_state)
        values = np.matmul(self.output_parameters.T, activation)

        return values

    def compute_activation(self, processed_state):
        """[TODO] Compute the action values values.
        Given a processed state, first we need to compute the activtaion
        (the output of hidden layer). Then we compute the values (the output of
        the output layer).
        """
        activation = np.matmul(self.hidden_parameters.T, processed_state)

        return activation

    def compute_gradient(self, processed_states, actions, rewards, tau, T):
        n = self.n

        # [TODO] compute the target value.
        # Hint: copy your codes in LinearTrainer.
        valid_states = processed_states[-n-1:]
        valid_rewards = rewards[-n:]

        q = sum([reward * self.gamma ** i for i, reward in enumerate(valid_rewards)])
        G = np.zeros((self.act_dim, 1))
        action = actions[-n-1]
        G[action,0] = q

        if tau + n < T:
            next_n_values = self.compute_values(valid_states[-1])
            G += self.gamma ** n * next_n_values[:, None]

        # Denote the state-action value function Q, then the loss of
        # prediction error w.r.t. the output layer weights can be
        # separated into two parts (the chain rule):
        #     dError / dweight = (dError / dQ) * (dQ / dweight)
        # We call the first one loss_grad, and the latter one
        # value_grad. We consider the Mean Square Error between the target
        # value (G) and the predict value (Q(s_t, a_t)) to be the loss.
        # current_state = processed_states[tau]

        loss_grad = np.zeros((self.act_dim, 1))  # [act_dim, 1]
        # [TODO] compute loss_grad
        current_state = valid_states[0]
        current_q = self.compute_values(current_state)[:, None]
        loss_grad = G - current_q

        # [TODO] compute the gradient of output layer parameters
        output_gradient = np.matmul(np.matmul(self.hidden_parameters.T, current_state[:, None]), loss_grad.T)

        # [TODO] compute the gradient of hidden layer parameters
        hidden_gradient = np.matmul(np.matmul(current_state[:,None], loss_grad.T), self.output_parameters.T)

        assert np.all(np.isfinite(output_gradient)), \
            "Invalid value occurs in output_gradient! {}".format(
                output_gradient)
        assert np.all(np.isfinite(hidden_gradient)), \
            "Invalid value occurs in hidden_gradient! {}".format(
                hidden_gradient)
        return [hidden_gradient, output_gradient]

    def apply_gradient(self, gradients):
        """Apply the gradientss to the two layers' parameters."""
        assert len(gradients) == 2
        hidden_gradient, output_gradient = gradients

        assert output_gradient.shape == (self.hidden_dim, self.act_dim)
        assert hidden_gradient.shape == (self.obs_dim, self.hidden_dim)

        # [TODO] Implement the clip gradient mechansim
        # Hint: when the old gradient has norm less that clip_norm,
        #  then nothing happens. Otherwise shrink the gradient to
        #  make its norm equal to clip_norm.
        if self.config["clip_gradient"]:
            clip_norm = self.config["clip_norm"]
            output_gradient = output_gradient * clip_norm / max(np.linalg.norm(output_gradient), clip_norm)
            hidden_gradient = hidden_gradient * clip_norm / max(np.linalg.norm(hidden_gradient), clip_norm)

        # [TODO] update the parameters
        # Hint: Remember to check the sign when applying the gradient
        #  into the parameters. Should you add or minus the gradients?
        self.hidden_parameters += self.learning_rate * hidden_gradient
        self.output_parameters += self.learning_rate * output_gradient

if __name__ == '__main__':
    print("Now let's see what happen if gradient clipping is enable!\n")
    mlp_trainer, _ = run(MLPTrainer, dict(
        max_iteration=3000,
        evaluate_interval=100,
        parameter_std=0.01,
        learning_rate=0.001,
        hidden_dim=100,
        clip_gradient=True,  # <<< Gradient clipping is ON!
        env_name="CartPole-v0"
    ), reward_threshold=195.0)