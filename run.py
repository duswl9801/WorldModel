import os
import torch
from utils import *

def train(config, env, agent, replay_buffer, train_logger, run_dir):
    print("===== Training start =====")

    for step in range(1, config.train_steps + 1):
        for _ in range(config.wm_updates_per_step):
            batch = replay_buffer.sample_sequence_batch(config.batch_size)
            wm_metrics = agent.train_world_model(batch)
            train_logger.log_dict(wm_metrics)

        for _ in range(config.ac_updates_per_step):
            batch = replay_buffer.sample_sequence_batch(config.batch_size)
            ac_metrics = agent.train_actor_critic(batch)
            train_logger.log_dict(ac_metrics)

        collect_agent_steps(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            num_steps=config.collect_steps,
            crop_size=config.crop_size,
            deterministic=False,
        )

        if step % config.log_every == 0:
            train_logger.write(step=step)

        if step % config.save_every == 0:
            ckpt_path = os.path.join(run_dir, f"checkpoint_step_{step}.pt")
            agent.save_checkpoint(ckpt_path, step=step)
            print(f"saved: {ckpt_path}")

    final_ckpt = os.path.join(run_dir, "checkpoint_final.pt")
    agent.save_checkpoint(final_ckpt, step=config.train_steps)
    print(f"Training finished. Final checkpoint saved to: {final_ckpt}")


@torch.no_grad()
def eval(config, env, agent, eval_logger=None, step=None, save_frames=False):
    metrics, frames = evaluate_agent(
        env=env,
        agent=agent,
        crop_size=config.crop_size,
        num_episodes=config.eval_episodes,
        save_frames=save_frames
    )

    if eval_logger is not None:
        eval_logger.log_dict(metrics)
        if step is None:
            eval_logger.write()
        else:
            eval_logger.write(step=step)

    print(metrics)
    return metrics, frames


def train_eval(config, env, eval_env, agent, replay_buffer, train_logger, eval_logger, run_dir, save_frames=False):
    print("===== Training + Evaluation start =====")

    for step in range(1, config.train_steps + 1):
        for _ in range(config.wm_updates_per_step):
            batch = replay_buffer.sample_sequence_batch(config.batch_size)
            wm_metrics = agent.train_world_model(batch)
            train_logger.log_dict(wm_metrics)

        for _ in range(config.ac_updates_per_step):
            batch = replay_buffer.sample_sequence_batch(config.batch_size)
            ac_metrics = agent.train_actor_critic(batch)
            train_logger.log_dict(ac_metrics)

        collect_agent_steps(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            num_steps=config.collect_steps,
            crop_size=config.crop_size,
            deterministic=False,
        )

        if step % config.log_every == 0:
            train_logger.write(step=step)

        if step % config.eval_every == 0:
            eval_metrics, _ = evaluate_agent(
                env=eval_env,
                agent=agent,
                crop_size=config.crop_size,
                num_episodes=config.eval_episodes,
                save_frames=False,
            )
            eval_logger.log_dict(eval_metrics)

            report_batch = replay_buffer.sample_sequence_batch(config.batch_size)
            report = agent.report(report_batch)
            eval_logger.log_dict(report["metrics"])

            eval_logger.write(step=step)

        if step % config.save_every == 0:
            ckpt_path = os.path.join(run_dir, f"checkpoint_step_{step}.pt")
            agent.save_checkpoint(ckpt_path, step=step)
            print(f"saved: {ckpt_path}")

    final_ckpt = os.path.join(run_dir, "checkpoint_final.pt")
    agent.save_checkpoint(final_ckpt, step=config.train_steps)
    print(f"Training finished. Final checkpoint saved to: {final_ckpt}")

    final_metrics, final_frames = evaluate_agent(
        env=eval_env,
        agent=agent,
        crop_size=config.crop_size,
        num_episodes=config.eval_episodes,
        save_frames=save_frames,
    )
    print("Final eval:", final_metrics)
    return final_metrics, final_frames