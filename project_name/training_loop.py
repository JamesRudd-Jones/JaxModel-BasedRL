import jax
import jax.numpy as jnp
import jax.random as jrandom
from project_name.utils import PlotTuple, make_normalised_plot_fn
from project_name import utils
import sys
import logging
from functools import partial
import time
from project_name.utils_plots import plot, plotters
from jaxtyping import Float, install_import_hook
from jax.experimental import checkify
import bifurcagym

with install_import_hook("gpjax", "beartype.beartype"):
    # import logging
    # logging.getLogger('gpjax').setLevel(logging.WARNING)  # prevents loads of gpjax logging
    import gpjax


def run_train(config):
    key = jrandom.key(config.SEED)

    env = bifurcagym.make(config.ENV_NAME,
                          cont_state=True,
                          cont_action=True,
                          vmappable=False,
                          normalised=config.NORMALISE_ENV,
                          autoreset=True,
                          metrics=True,
                          )

    # add plot functionality as required
    plot_fn = partial(plotters[config.ENV_NAME], env=env)  # TODO sort this out
    if config.NORMALISE_ENV:
        plot_fn = make_normalised_plot_fn(env, plot_fn)

    start_obs, start_env_state = utils.get_start_obs(env, key)  # TODO set a consistent start point defined in config

    actor = utils.import_class_from_folder(config.AGENT_TYPE)(env=env, config=config, key=key)

    # Initial data for training/system identification. This should be agent specific
    sys_id_data = utils.get_initial_data(config, utils.get_f_generative, plot_fn, env,
                                                          actor.agent_config.SYS_ID_DATA, key, train=True)
    # key, _key = jrandom.split(key)
    # test_data = utils.get_initial_data(config, utils.get_f_generative, plot_fn, env, config.NUM_INIT_DATA, _key)

    key, _key = jrandom.split(key)
    if config.PRETRAIN_HYPERPARAMS:
        pretrain_data = utils.get_initial_data(config, utils.get_f_generative, plot_fn, env, config.PRETRAIN_NUM_DATA,
                                               _key)
        train_state = actor.pretrain_params(sys_id_data, pretrain_data, _key)
    else:
        train_state = actor.create_train_state(sys_id_data, _key)
        # TODO how does the above work if there is no data, can we use the start obs and a random action or nothing?

    # Ground truth data for evaluation
    if actor.agent_config.ROLLOUT_SAMPLING:
        key, _key = jrandom.split(key)
        batch_key = jrandom.split(_key, config.NUM_EVAL_TRIALS)
        start_gt_time = time.time()
        true_paths, test_points, path_lengths, all_returns = (jax.vmap(actor.execute_gt_mpc,
                                                                       in_axes=(None, None, None, None, 0))
                                                              (start_obs,
                                                               utils.get_f_generative,
                                                               train_state,
                                                               (sys_id_data.X, sys_id_data.y),
                                                               batch_key))
        logging.info(f"Ground truth time taken = {time.time() - start_gt_time:.2f}s; "
                     f"Mean Return = {jnp.mean(all_returns):.2f}; Std Return = {jnp.std(all_returns):.2f}; "
                     f"Mean Path Lengths = {jnp.mean(path_lengths)}; ")

        # Plot ground truth paths and print info
        utils.make_plots(plot_fn, true_paths, None, env, config, actor.agent_config, None, None, None, None, gt=True)
    else:
        true_paths = {"exe_path_x": jnp.zeros((1, 10, 3)),
                      "exe_path_y": jnp.zeros((1, 10, 2))}
        # TODO how can we give some groundtruth even if not using MPC?

    # Run the main learning loop
    def _main_loop(curr_obs_O, train_data, train_state, key):
        global_returns = 0
        for step_idx in range(0, config.NUM_ITERS):
            step_start_time = time.time()
            logging.info("---" * 5 + f" Start iteration i={step_idx} " + "---" * 5)
            logging.info(f"Length of data.x: {len(train_data.X)}; Length of data.y: {len(train_data.y)}")

            # Get next point
            err, get_next_point_output = checkify.checkify(actor.get_next_point)(curr_obs_O, train_state, train_data,
                                                                                 step_idx, key)
            err.throw()
            x_next_OPA, exe_path, curr_obs_O, train_state, acq_val, key = get_next_point_output

            # Periodically run evaluation and plot
            if (step_idx % config.EVAL_FREQ == 0 or step_idx + 1 == config.NUM_ITERS):
                key, _key = jrandom.split(key)
                batch_key = jrandom.split(_key, config.NUM_EVAL_TRIALS)
                start_obs, start_env_state = utils.get_start_obs(env, key)
                eval_output = jax.vmap(actor.evaluate, in_axes=(None, None, None, None, 0))(start_obs,
                                                                                            start_env_state,
                                                                                            train_state,
                                                                                            (train_data.X, train_data.y),
                                                                                            batch_key)
                real_paths_mpc, real_returns, mean_returns, std_returns, mse = eval_output
                logging.info(f"Eval Returns = {real_returns}; Mean = {jnp.mean(mean_returns):.2f}; "
                             f"Std = {jnp.std(std_returns):.2f}")  # TODO check the std
                logging.info(f"Model MSE = {jnp.mean(mse):.2f}")

                # TODO add testing on the random test_data that we created initially

                utils.make_plots(plot_fn,
                                 PlotTuple(x=true_paths["exe_path_x"][-1], y=true_paths["exe_path_y"][-1]),
                                 PlotTuple(x=train_data.X, y=train_data.y),
                                 env, config, actor.agent_config, exe_path, real_paths_mpc, x_next_OPA,
                                 step_idx)

            action_A = x_next_OPA[-env.action_space().shape[0]:]

            # Query function, update data
            key, _key = jrandom.split(key)
            y_next_O = utils.get_f_generative(jnp.expand_dims(x_next_OPA, axis=0), env, train_state, train_data, _key)

            if actor.agent_config.ROLLOUT_SAMPLING:
                delta = y_next_O[-env.observation_space().shape[0]:]
                nobs_O = actor._update_fn(curr_obs_O, delta, env)
                # TODO sort the above out, it works when curr_obs doesn't change
            else:
                delta = y_next_O[-env.observation_space().shape[0]:]
                nobs_O = actor._update_fn(curr_obs_O, delta, env)

                # # raise NotImplementedError("When is it not rollout sampling?")
                # # TODO dodgy fix for now
                # nobs_O, new_env_state, reward, done, info = env.step(_key, env_state, action_A)
                # y_next_O = nobs_O - curr_obs_O
                # TODO this does not work with periodic envs and was causing issues
                #
                # global_returns += reward

            logging.info(f"Iteration i={step_idx} Action: {action_A}")
            logging.info(f"Iteration i={step_idx} State : {nobs_O}")
            logging.info(f"iteration i={step_idx} Return so far: {global_returns}")

            train_data = train_data + gpjax.Dataset(X=jnp.expand_dims(x_next_OPA, axis=0),
                                                    y=jnp.expand_dims(y_next_O, axis=0))
            # TODO will the above work with PETS as well?
            curr_obs_O = nobs_O
            logging.info(f"Iteration time taken - {time.time() - step_start_time:.1f}s")

    _main_loop(start_obs, sys_id_data, train_state, key)

    return
