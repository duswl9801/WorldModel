import gymnasium as gym
import matplotlib.pyplot as plt


def main():
    env = gym.make("highway-v0", render_mode="rgb_array")

    obs, info = env.reset()

    print("Observation type:", type(obs))
    print("Observation shape:", obs.shape if hasattr(obs, "shape") else "no shape")
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("Info keys:", info.keys())

    frame = env.render()
    print("Rendered frame shape:", frame.shape)

    plt.imshow(frame)
    plt.title("highway-v0 first frame")
    plt.axis("off")
    plt.show()

    env.close()


if __name__ == "__main__":
    main()