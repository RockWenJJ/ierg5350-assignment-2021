from core.utils import load_progress
import matplotlib.pyplot as plt
import seaborn as sns

##### draw single progress
ppo_progress = load_progress("MetaDriveEasy/A2C")
plt.figure(dpi=300)
sns.set("notebook", "darkgrid")
ax = sns.lineplot(
    data=ppo_progress,
    x="total_steps",
    y="training_episode_reward/episode_reward_mean",
    # y="success_rate/success_rate_mean"
)
ax.set_title("A2C training result in MetaDriveEasy")
ax.set_ylabel("Episode Reward Mean")
ax.set_xlabel("Sampled Steps")
# ax.set_ylabel("Success Rate Mean")
# plt.xlim(0, 1300000)
plt.show()

##### draw generalized results
# ### ppo generalized results, data from recorded stats.txt
# train_success_rates = [1.0, 0.86, 0.82, 0.74, 0.63, 0.63]
# test_success_rates = [0.1, 0.52, 0.54, 0.50, 0.55, 0.65]
# envs = ['1', '5', '10', '20', '50', '100']
# plt.plot(envs, train_success_rates)
# plt.plot(envs, test_success_rates)
# plt.ylim([0., 1.1])
# plt.legend(["training", "Testing"])
# plt.xlabel("Training Scenes")
# plt.ylabel("Mean Success Rate")
# plt.title("PPO Generalized Results")
# plt.grid()
# plt.show()
# ### td3 generalized results, data from recorded stats.txt
# train_success_rates = [1.0, 0.98, 0.98, 0.80, 0.83, 0.80]
# test_success_rates = [0.38, 0.72, 0.74, 0.54, 0.72, 0.87]
# envs = ['1', '5', '10', '20', '50', '100']
# plt.plot(envs, train_success_rates)
# plt.plot(envs, test_success_rates)
# plt.ylim([0., 1.1])
# plt.legend(["training", "Testing"])
# plt.xlabel("Training Scenes")
# plt.ylabel("Mean Success Rate")
# plt.title("TD3 Generalized Results")
# plt.grid()
# plt.show()
