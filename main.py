import os
import datetime
from pathlib import Path

import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from agent import Agent
from wrappers import SkipFrame, ResizeObservation

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == "__main__":
    # スーパーマリオの環境を初期化
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

    # 行動空間を以下に制限
    #   0. 右に歩く
    #   1. 右方向にジャンプ
    env = JoypadSpace(
        env,
        [['right'],
         ['right', 'A']]
    )

    # 環境にWrapperを適用
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)

    env.reset()

    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    checkpoint = None  # Path('trained_mario.chkpt')
    mario = Agent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

    episodes = 10

    for e in range(episodes):
        total_reward = 0
        step = 0
        state = env.reset()

        while True:
            step += 1
            env.render()

            action = mario.act(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            mario.cache(state, next_state, action, reward, done)

            q, loss = mario.learn()
            state = next_state

            if done or info['flag_get']:
                print(f"Episode {e + 1}/{episodes} finished after {step} episode steps with total reward = {total_reward}")
                break

    mario.save()
