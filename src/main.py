from warnings import catch_warnings

with catch_warnings():
    import compiler_gym
    import gym


def main():
    print(compiler_gym.COMPILER_GYM_ENVS)
    env = gym.make("llvm-v0")
    print(list(env.datasets))
    env.reset(env.datasets["benchmark://jotaibench-v0"].random_benchmark())
    print(env.observation["IsRunnable"])
    print(compiler_gym.envs.llvm.datasets.JotaiBenchRunnableDataset.__dict__)
    # print(env.render())


if __name__ == "__main__":
    main()
