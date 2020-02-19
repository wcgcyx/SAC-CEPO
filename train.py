import sys
import sac
import sac_cepo
from utils import get_normalized_env


def train():
    agent_name = sys.argv[1]
    N = int(sys.argv[2])
    Ne = int(sys.argv[3])
    size = float(sys.argv[4])
    t = int(sys.argv[5])
    task = sys.argv[6]
    reward_scale = int(sys.argv[7])
    max_step = int(sys.argv[8])
    evaluate_step = int(sys.argv[9])
    output_file = sys.argv[10]
    print("Agent: {} with {} {} {} {}".format(agent_name, N, Ne, size, t))
    # Load environment and agent
    env = get_normalized_env(task)
    eval_env = get_normalized_env(task)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if agent_name == 'sac':
        agent = sac.Agent(state_dim=state_dim, action_dim=action_dim, alpha=1 / reward_scale)
    else:
        agent = sac_cepo.Agent(state_dim=state_dim, action_dim=action_dim, alpha=1 / reward_scale,
                               ce_n=N, ce_ne=Ne, ce_size=size, ce_t=t)

    # Initial exploration for 1000 steps
    step = 0
    while True:
        state = env.reset()
        while True:
            action = env.action_space.sample()
            next_state, reward, end, _ = env.step(action)
            step += 1
            agent.store_transition(state, action, reward, next_state, end * 1)
            state = next_state
            if end:
                break
            if step == 1000:
                break
        if step == 1000:
            break

    # Formal training
    step = 0
    while True:
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, end, _ = env.step(action)
            step += 1
            agent.store_transition(state, action, reward, next_state, end * 1)
            agent.learn()
            state = next_state
            if step % evaluate_step == 0:
                evaluate_reward = rollout(agent, eval_env)
                print(step, evaluate_reward)
                with open(output_file, "a") as file:
                    file.write("{},{}\n".format(step, evaluate_reward))
            if end:
                break
            if step == max_step:
                break
        if step == max_step:
            break

    env.close()
    eval_env.close()
    print("Training finished.")


def rollout(agent, env):
    average_reward = 0
    for i in range(10):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.get_rollout_action(state)
            next_state, reward, end, _ = env.step(action)
            state = next_state
            total_reward += reward
            if end:
                break
        average_reward += total_reward
    return average_reward / 10


if __name__ == "__main__":
    if len(sys.argv) != 11:
        print("Usage: py -3 train.py agent_name N Ne size t task reward_scale max_step evaluate_step output_file")
        exit(1)
    train()
