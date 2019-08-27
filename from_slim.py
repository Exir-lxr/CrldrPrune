
import tensorflow as tf

def sum_clones_gradients(clone_grads):
  """Calculate the sum gradient for each shared variable across all clones.

  This function assumes that the clone_grads has been scaled appropriately by
  1 / num_clones.

  Args:
    clone_grads: A List of List of tuples (gradient, variable), one list per
    `Clone`.

  Returns:
     List of tuples of (gradient, variable) where the gradient has been summed
     across all clones.
  """
  sum_grads = []
  for grad_and_vars in zip(*clone_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
    grads = []
    var = grad_and_vars[0][1]
    for g, v in grad_and_vars:
      assert v == var
      if g is not None:
        grads.append(g)
    if grads:
      if len(grads) > 1:
        sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
      else:
        sum_grad = grads[0]
      sum_grads.append((sum_grad, var))
  return sum_grads
