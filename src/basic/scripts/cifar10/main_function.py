import tensorflow as tf
import os

from basic.scripts.cifar10.config import RunConfig
from basic.scripts.cifar10.model_function import get_experiment_fn, get_estimator_data


def main_new(job_dir, data_dir, num_gpus, variable_strategy,
             use_distortion_for_training, log_device_placement,
             num_intra_threads, **hparams):
    # The env variable is on deprecation path, default is set to off.
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Session configuration.
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=log_device_placement,
        intra_op_parallelism_threads=num_intra_threads,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    config = RunConfig(
        session_config=sess_config, model_dir=job_dir)
    estimator_data = get_estimator_data(
        data_dir, num_gpus, variable_strategy,
        config, hparams, use_distortion_for_training=use_distortion_for_training)
    estimator = estimator_data.get('classifier')
    train_spec = tf.estimator.TrainSpec(input_fn=estimator_data.get('train_input_fn'),
                                        max_steps=estimator_data.get('train_steps'))
    eval_spec = tf.estimator.EvalSpec(input_fn=estimator_data.get('eval_input_fn'),
                                      steps=estimator_data.get('eval_steps'))
    return tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(job_dir, data_dir, num_gpus, variable_strategy,
         use_distortion_for_training, log_device_placement, num_intra_threads,
         **hparams):
    # The env variable is on deprecation path, default is set to off.
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Session configuration.
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=log_device_placement,
        intra_op_parallelism_threads=num_intra_threads,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    config = RunConfig(
        session_config=sess_config, model_dir=job_dir)
    tf.contrib.learn.learn_runner.run(
        get_experiment_fn(data_dir, num_gpus, variable_strategy,
                          use_distortion_for_training),
        run_config=config,
        hparams=tf.contrib.training.HParams(
            is_chief=config.is_chief,
            **hparams))
