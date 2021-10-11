import gym
import numpy as np
from collections import deque
from utils import *
from trainer import *

default_pi_config = dict(
    max_iteration=1000,
    evaluate_interval=1,
    gamma=1.0,
    eps=1e-10
)


def policy_iteration(train_config=None):
    config = default_pi_config.copy()
    if train_config is not None:
        config.update(train_config)

    trainer = PolicyItertaionTrainer(gamma=config['gamma'], eps=config['eps'])

    #     old_policy_result = {
    #         obs: -1 for obs in range(trainer.obs_dim)
    #     }
    old_policy_table = -1 * np.ones([trainer.obs_dim, ], dtype=np.int)

    for i in range(config['max_iteration']):
        # train the agent
        trainer.train()  # [TODO] please uncomment this line

        # [TODO] compare the new policy with old policy to check whether
        #  should we stop. If new and old policy have same output given any
        #  observation, them we consider the algorithm is converged and
        #  should be stopped.
        should_stop = np.all((trainer.policy_table - old_policy_table) == 0)

        if should_stop:
            print("We found policy is not changed anymore at "
                  "itertaion {}. Current mean episode reward "
                  "is {}. Stop training.".format(i, trainer.evaluate()))
            break
        old_policy_table = trainer.policy_table

        # evaluate the result
        if i % config['evaluate_interval'] == 0:
            print(
                "[INFO]\tIn {} iteration, current mean episode reward is {}."
                "".format(i, trainer.evaluate()))

            if i > 20:
                print("You sure your codes is OK? It shouldn't take so many "
                      "({}) iterations to train a policy iteration "
                      "agent.".format(i))

    assert trainer.evaluate() > 0.8, \
        "We expect to get the mean episode reward greater than 0.8. " \
        "But you get: {}. Please check your codes.".format(trainer.evaluate())

    return trainer


if __name__ == '__main__':
    pi_agent = policy_iteration()