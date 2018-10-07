from basic.scripts.slim import dataset_utils, model_deploy
import tensorflow as tf
from basic.scripts.slim import inception_v1
from basic.scripts.slim import inception_preprocessing

num_clones = 1
clone_on_cpu = True
task = 0
worker_replicas = 1
num_ps_tasks = 0
split_name = 'training'
num_samples = 39200
num_classes = 1001
output_dir = '/home/jdavidagudelo/Documents/data_sets/minist/tf_records/'
batch_size = 32
num_readers = 4
labels_offset = 0
num_preprocessing_threads = 4
label_smoothing = 0.0
moving_average_decay = None
quantize_delay = -1
sync_replicas = False
replicas_to_aggregate = 1
train_dir = '/home/jdavidagudelo/Documents/data_sets/slim_models/inception_v1/'
master = ''
max_number_of_steps = None
log_every_n_steps = 10
save_summaries_secs = 300
save_interval_secs = 300
num_epochs_per_decay = 2.0
learning_rate = 0.01
learning_rate_decay_factor = 0.94
end_learning_rate = 0.0001
learning_rate_decay_type = 'exponential'
optimizer = 'adadelta'
opt_epsilon = 1.0
adadelta_rho = 0.95
adagrad_initial_accumulator_value = 0.1
adam_beta1 = 0.9
adam_beta2 = 0.999
trainable_scopes = None
checkpoint_path = '/home/jdavidagudelo/Documents/data_sets/slim_models/inception_v1/inception_v1.ckpt'
checkpoint_exclude_scopes = None
ignore_missing_vars = True
momentum = 0.9
rmsprop_momentum = 0.9
rmsprop_decay = 0.9
ftrl_learning_rate_power = -0.5
ftrl_initial_accumulator_value = 0.1
ftrl_l2 = 0.0
ftrl_l1 = 0.0
train_image_size = 224

slim = tf.contrib.slim


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
    # Note: when num_clones is > 1, this will actually have each clone to go
    # over each epoch FLAGS.num_epochs_per_decay times. This is different
    # behavior from sync replicas and is expected to produce different results.
    decay_steps = int(num_samples_per_epoch * num_epochs_per_decay /
                      batch_size)

    if sync_replicas:
        decay_steps /= replicas_to_aggregate

    if learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(learning_rate,
                                          global_step,
                                          decay_steps,
                                          learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif learning_rate_decay_type == 'fixed':
        return tf.constant(learning_rate, name='fixed_learning_rate')
    elif learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(learning_rate,
                                         global_step,
                                         decay_steps,
                                         end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                         learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
    """
    if optimizer == 'adadelta':
        result = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=adadelta_rho,
            epsilon=opt_epsilon)
    elif optimizer == 'adagrad':
        result = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=adagrad_initial_accumulator_value)
    elif optimizer == 'adam':
        result = tf.train.AdamOptimizer(
            learning_rate,
            beta1=adam_beta1,
            beta2=adam_beta2,
            epsilon=opt_epsilon)
    elif optimizer == 'ftrl':
        result = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=ftrl_learning_rate_power,
            initial_accumulator_value=ftrl_initial_accumulator_value,
            l1_regularization_strength=ftrl_l1,
            l2_regularization_strength=ftrl_l2)
    elif optimizer == 'momentum':
        result = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=momentum,
            name='Momentum')
    elif optimizer == 'rmsprop':
        result = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=rmsprop_decay,
            momentum=rmsprop_momentum,
            epsilon=opt_epsilon)
    elif optimizer == 'sgd':
        result = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized' % optimizer)
    return result


def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
    if checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % train_dir)
        return None

    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(checkpoint_path):
        result = tf.train.latest_checkpoint(checkpoint_path)
    else:
        result = checkpoint_path

    tf.logging.info('Fine-tuning from %s' % result)

    return slim.assign_from_checkpoint_fn(
        result,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)


def _get_variables_to_train():
    """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
    if trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main():
    with tf.Graph().as_default():
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=num_clones,
            clone_on_cpu=clone_on_cpu,
            replica_id=task,
            num_replicas=worker_replicas,
            num_ps_tasks=num_ps_tasks)
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        dataset = dataset_utils.get_split(split_name, num_samples, num_classes,
                                          output_dir)
        network_fn = inception_v1.inception_v1

        with tf.device(deploy_config.inputs_device()):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=num_readers,
                common_queue_capacity=20 * batch_size,
                common_queue_min=10 * batch_size)
            [image, label] = provider.get(['image', 'label'])
            label -= labels_offset
            image = inception_preprocessing.preprocess_image(image, train_image_size, train_image_size, is_training=True)
            images, labels = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocessing_threads,
                capacity=5 * batch_size)
            labels = slim.one_hot_encoding(
                labels, dataset.num_classes - labels_offset)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2 * deploy_config.num_clones)

        def clone_fn(batch_queue):
            """Allows data parallelism by creating multiple clones of network_fn."""
            imgs, ls = batch_queue.dequeue()
            logits, end_points = network_fn(imgs, num_classes=num_classes)

            #############################
            # Specify the loss function #
            #############################
            if 'AuxLogits' in end_points:
                slim.losses.softmax_cross_entropy(
                    end_points['AuxLogits'], ls,
                    label_smoothing=label_smoothing, weights=0.4,
                    scope='aux_loss')
            slim.losses.softmax_cross_entropy(
                logits, ls, label_smoothing=label_smoothing, weights=1.0)
            return end_points

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        #################################
        # Configure the moving averages #
        #################################
        if moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        if quantize_delay >= 0:
            tf.contrib.quantize.create_training_graph(
                quant_delay=quantize_delay)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            lr = _configure_learning_rate(dataset.num_samples, global_step)
            current_optimizer = _configure_optimizer(lr)
            summaries.add(tf.summary.scalar('learning_rate', lr))

        if sync_replicas:
            # If sync_replicas is enabled, the averaging will be done in the chief
            # queue runner.
            current_optimizer = tf.train.SyncReplicasOptimizer(
                opt=current_optimizer,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=worker_replicas,
                variable_averages=variable_averages,
                variables_to_average=moving_average_variables)
        elif moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = _get_variables_to_train()

        #  and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            current_optimizer,
            var_list=variables_to_train)
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient updates.
        grad_updates = current_optimizer.apply_gradients(clones_gradients,
                                                         global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        slim.learning.train(
            train_tensor,
            logdir=train_dir,
            master=master,
            is_chief=(task == 0),
            init_fn=_get_init_fn(),
            summary_op=summary_op,
            number_of_steps=max_number_of_steps,
            log_every_n_steps=log_every_n_steps,
            save_summaries_secs=save_summaries_secs,
            save_interval_secs=save_interval_secs,
            sync_optimizer=current_optimizer if sync_replicas else None)
