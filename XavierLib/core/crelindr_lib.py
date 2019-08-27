import tensorflow as tf
import numpy as np

CONFIG_NUM = 1e-4
INF = 1e10


class MinimumContainer(object):

    def __init__(self, length):
        """

        self.content: { ... value: [properties], ...}
        """
        self.max_len = length
        self.content = {np.inf: None}

    def push(self, value, properties):
        if len(self.content) == self.max_len:
            max_in_content = max(self.content)
            if value < max(self.content):
                del self.content[max_in_content]
                self.content[value] = properties
        elif len(self.content) < self.max_len:
            self.content[value] = properties
        else:
            raise Exception('TOO MUCH!', self.max_len)


# Extra-part
def low_precision(tensor):
    low = tf.cast(tensor, tf.float16)
    return tf.cast(low, tf.float64)


# PART 1: Statistic
class accu_return(object):
    def __init__(self):
        self.op_list = []
        self.variable_avg = None
        self.summary = None
        self.count = None
        self.name = None


def accumulate(var, name, precision=tf.float32):
    '''
    Builds flows to compute the mean of the var.
    :param var:         A tensor, the variable to compute the mean value.
    :param name:        A string, the name.
    :return:            op_list:        A list of op tensor, run which to update var to count and summary.
                        variable_avg:   A tensor, the mean value.
                        summary:        A variable, the total sum.
                        count:          A variable, the total amount.
                        name:           A string, the name.
    '''
    a = accu_return()
    a.name = name
    with tf.variable_scope(name):
        count = tf.get_variable('count', [], precision, tf.zeros_initializer, trainable=False)
        a.count = count
        a.op_list.append(tf.assign(count, tf.add(count, 1)))

        count = tf.maximum(count, tf.constant(1, precision, []))

        summary = tf.get_variable('sum', var.shape, precision, tf.zeros_initializer, trainable=False)
        a.summary = summary
        a.op_list.append(tf.assign(summary, tf.add(summary, var)))
        a.variable_avg = tf.divide(summary, count)
    return a


class statistic_return(object):
    def __init__(self):
        self.op_list = []
        self.name = None
        self.zero_var_mask_one = None
        self.zero_var_mask_zero = None
        self.zero_var_mask_non_zero = None
        self.mean = None
        self.var = None
        self.covariance = None
        self.coe = None
        self.k = None
        self.acls_list = []

    def get_source(self):
        return_list = []
        for acls in self.acls_list:
            return_list.append(acls.summary)
            return_list.append(acls.count)
        return return_list

    def get_reset_op_list(self):
        return_op_list = []
        for acls in self.acls_list:
            return_op_list.append(tf.assign(acls.summary, tf.constant(0, tf.float64, acls.summary.get_shape().as_list())))
            return_op_list.append(tf.assign(acls.count, tf.constant(0, tf.float64, acls.count.get_shape().as_list())))
        return return_op_list


class statistic_flow_builder(object):
    def __init__(self):
        self.scls_list = []

    def build_statistic_flow_on_conv_inputs(self, input_data, kshape=[1, 1], strides=1, padding='SAME'):

        with tf.variable_scope('forward_statistic'):
            a = statistic_return()

            input_data = low_precision(input_data)

            if kshape == [1, 1]:
                m = tf.reduce_mean(input_data, [0, 1, 2])
                centralized_x = input_data - m
            else:
                # set stride=1 can preventing over-fitting
                patch_input_data = tf.extract_image_patches(input_data, [1, kshape[0], kshape[1], 1],
                                                            [1, strides, strides, 1],
                                                            [1, 1, 1, 1], padding)
                m = tf.reduce_mean(patch_input_data, [0, 1, 2])
                centralized_x = patch_input_data - m
            s_num = centralized_x.get_shape().as_list()[0] * centralized_x.get_shape().as_list()[1] * \
                    centralized_x.get_shape().as_list()[2]
            samples = tf.reshape(centralized_x, [s_num, centralized_x.get_shape().as_list()[3]])

            v = tf.matmul(tf.transpose(samples), samples) / s_num
            m_acls = accumulate(m, 'inputs_means', tf.float64)
            v_acls = accumulate(v, 'inputs_covars', tf.float64)
            a.acls_list += [m_acls, v_acls]
            a.op_list += m_acls.op_list
            a.op_list += v_acls.op_list
            a.mean = m_acls.variable_avg
            a.covariance = v_acls.variable_avg

            var = tf.diag_part(v_acls.variable_avg)
            a.var = var
            sss = var.get_shape().as_list()[0]
            xx = tf.reshape(var, [sss])

            max_var = tf.reduce_max(xx)
            find_near_zero = tf.sign(xx - CONFIG_NUM * max_var)
            a.zero_var_mask_zero = find_near_zero
            a.zero_var_mask_one = 1 - find_near_zero
            a.zero_var_mask_non_zero = max_var / 10 * a.zero_var_mask_one

            #xx = xx + (1 - find_zero)
            #rsv = tf.rsqrt(xx) * find_zero
            #div_m = tf.matmul(tf.reshape(rsv, [sss, 1]), tf.reshape(rsv, [1, sss]))
            #coe = v_acls.variable_avg * div_m
            #a.coe = coe

            #k = tf.divide(tf.transpose(v_acls.variable_avg), xx)
            #k = tf.transpose(tf.multiply(k, find_zero))
            #k = tf.multiply(k, find_zero)
            #a.k = k

            stri = tf.get_variable_scope().name
            a.name = stri
            self.scls_list.append(a)
        return a

    def get_reset_op(self):
        op_list = []
        for sr in self.scls_list:
            op_list += sr.get_reset_op_list()
        return op_list

    def get_source_variables(self):
        var_list = []
        for sr in self.scls_list:
            var_list += sr.get_source()
        return var_list

    def get_update_op(self):
        op_list = []
        for sr in self.scls_list:
            op_list += sr.op_list
        return op_list


class average_gradients_flow_builder(object):
    def __init__(self):
        self.xs_list = []
        self.xs_properties_list = []
        self.y = None
        self.built_flag = False
        self.grad_list = []
        self.acls_list = []
        self.avg_grad_list = []
        self.built_method = None
        self.return_list = []

    def collect_xs(self, a_x, kshape=[1,1], stride=1, padding='SAME'):
        self.xs_list.append(a_x)
        self.xs_properties_list.append([kshape, stride, padding])

    def set_y(self, y):
        """
        :param y:   A tensor, must be a scale!!!
        """
        self.y = y

    def build_flow_on_conv(self, method='VAR'):
        with tf.variable_scope('backward_statistic'):
            if (len(self.xs_list) > 0) and (self.y is not None) and (not self.built_flag):
                self.grad_list = tf.gradients(self.y, self.xs_list)
                for ind in range(len(self.grad_list)):
                    # patch-wise
                    grad = self.grad_list[ind]
                    name = grad.name[:-2] + '_accumulation'
                    properties = self.xs_properties_list[ind]
                    kshape = properties[0]
                    strides = properties[1]
                    padding = properties[2]
                    if kshape == [1, 1]:
                        not_centralized_x = grad
                        s_num = not_centralized_x.get_shape().as_list()[0] * not_centralized_x.get_shape().as_list()[1] * \
                                not_centralized_x.get_shape().as_list()[2]
                        samples = tf.reshape(not_centralized_x, [s_num, not_centralized_x.get_shape().as_list()[3]])
                    else:
                        # set stride=1 can preventing over-fitting
                        patch_input_data = tf.extract_image_patches(grad, [1, kshape[0], kshape[1], 1],
                                                                    [1, strides, strides, 1],
                                                                    [1, 1, 1, 1], padding)
                        not_centralized_x = patch_input_data
                        s_num = not_centralized_x.get_shape().as_list()[0] * not_centralized_x.get_shape().as_list()[1] * \
                                not_centralized_x.get_shape().as_list()[2]
                        samples = tf.reshape(not_centralized_x, [s_num, not_centralized_x.get_shape().as_list()[3]])

                    if method == 'MEAN':
                        to_accumulate = tf.reduce_mean(samples, [0])
                        self.built_method = 'MEAN'
                    elif method == 'VAR':
                        to_accumulate = tf.matmul(tf.transpose(samples), samples)
                        # to_accumulate = tf.reduce_mean(var_samples, [0])
                        self.built_method = 'VAR'
                    else:
                        to_accumulate = tf.reduce_mean(samples, [0])
                        self.built_method = 'MEAN'
                        print('ERROR: Parameter method has to be MEAN or VAR! Now using MEAN!!!')

                    acls = accumulate(to_accumulate, name)
                    self.acls_list.append(acls)
                    self.avg_grad_list.append(acls.variable_avg)

                self.built_flag = True
                print('Build gradient flow.')
            elif len(self.xs_list) == 0:
                print('Warning: No xs collected.')
            elif self.y is None:
                print('Warning: No y set.')
            elif self.built_flag:
                print('Warning: Already built before.')
            else:
                print('Error: Unknown error!')

    def get_update_op_list(self):
        op_list = []
        if self.built_flag:
            for acls in self.acls_list:
                op_list += acls.op_list
        else:
            print('Warning: Flow is not built yet.')
        return op_list

    def get_reset_op(self):
        return_op_list = []
        for acls in self.acls_list:
            return_op_list.append(tf.assign(acls.summary, tf.constant(0, tf.float32, acls.summary.get_shape().as_list())))
            return_op_list.append(tf.assign(acls.count, tf.constant(0, tf.float32, acls.count.get_shape().as_list())))
        return return_op_list

    def get_source_variables(self):
        var_list = []
        for acls in self.acls_list:
            var_list.append(acls.summary)
            var_list.append(acls.count)
        return var_list

    def post_process_and_get_tensor_list(self):
        with tf.variable_scope('tear_apart'):
            if len(self.return_list) == 0:
                if self.built_method == 'MEAN':
                    print('Get the tensor list of gradients\' means.')
                    for acls in self.acls_list:
                        self.return_list.append(acls.variable_avg)
                elif self.built_method == 'VAR':
                    print('Get the tensor list of vv, where vv*vv\' = E(GG\')')
                    for acls in  self.acls_list:
                        e, v = tf.self_adjoint_eig(acls.variable_avg)
                        vv = tf.multiply(v, tf.sqrt(tf.maximum(e, 0)))
                        # vv*vv' = acls.variable_avg
                        self.return_list.append(tf.transpose(vv))
        return self.return_list


# PART 2: Compute the variance after decorrelation
def variances_after_decorrelation_and_corresponding_transformations(cov_matrix, zero_var_mask_non_zero,
                                                                    zero_var_mask_zero):
    '''
    This function calculates the variance of each activation after decorrelation according covariance matrix

    :param cov_matrix:  a tensor shaped [n, n], containing the covariance matrix for n activations.
    :return:            a tensor shaped [n, ], the variance after decorrelation for each activation, and
                        a tensor shaped [n, n], the transformations stack by rows.
    '''
    with tf.variable_scope('var_after_decorrelation'):
        add_matrix = tf.diag(zero_var_mask_non_zero)
        cov_matrix = tf.transpose(tf.transpose(cov_matrix) * zero_var_mask_zero) * zero_var_mask_zero
        # tf.add_to_collection('fuckshow', cov_matrix)
        cov_matrix = cov_matrix + add_matrix
        tf.add_to_collection('show', cov_matrix)
        cov_reverse = tf.matrix_inverse(cov_matrix, adjoint=True)
        diag_part = tf.diag_part(cov_reverse)
        rows_divide_diag = tf.transpose(tf.divide(tf.transpose(cov_reverse), diag_part))
        result_matrix = tf.matmul(tf.matmul(rows_divide_diag, cov_matrix), tf.transpose(rows_divide_diag))
        variance_after_decorrelation = tf.diag_part(result_matrix)
        variance_after_decorrelation = tf.multiply(variance_after_decorrelation, zero_var_mask_zero)
    return variance_after_decorrelation, rows_divide_diag


def group_decorrelation(cov_matrix, group_index):
    '''
    For a given covariance matrix deducted for a vector_length random vector
    and a group index like [... i-th, j-th, k-th ...],
    return a transformation that the diagonal part is ones
    and makes the [... i-th, j-th, k-th ...] no related with each others and random variables no in the list.
    ---------------------------------
    | trans * cov_matrix * trans'   |
    _________________________________
    :param cov_matrix:      A tensor, sized [vector_length, vector_length], float
    :param group_index:     A tensor, sized [num, 1], int
    :return:                A tensor, sized [vector_length, vector_length], float
    '''
    vector_length = int(cov_matrix.get_shape().as_list()[0])
    num = group_index.get_shape().as_list()[0]

    index_proto = []
    for i in range(num):
        index_proto.append(i)
    index_proto = tf.Variable(index_proto)
    index_proto = tf.reshape(index_proto, [num, 1])
    sparse_index = tf.concat([index_proto, group_index], axis=1)

    take_matrix = tf.sparse_to_dense(sparse_index, [num, vector_length], 1.0)

    c_inv = tf.matrix_inverse(cov_matrix)

    c_sharp = tf.transpose(tf.matmul(tf.matmul(take_matrix, c_inv), tf.transpose(take_matrix)))

    _, m = tf.self_adjoint_eig(c_sharp)

    cm = tf.matmul(c_sharp, m)

    cm_diag = tf.diag_part(cm)
    cm_diag = tf.reshape(cm_diag, [num])

    m_ = tf.transpose(tf.divide(m, cm_diag))

    m_large = tf.matmul(tf.matmul(tf.transpose(take_matrix), m_), take_matrix)

    trans = tf.matmul(m_large, c_inv)

    one_diag = tf.diag_part(trans)
    all_one = tf.ones([vector_length], tf.float32)
    inv_diag = tf.subtract(all_one, one_diag)
    com_matrix = tf.diag(inv_diag)
    trans = tf.add(trans, com_matrix)

    return trans


def convolution_group_decorrelation(cov_matrix, kernel_size, channel_index):
    '''
    :param cov_matrix:      A tensor, sized [n, n] for n-dimensional random vector.
    :param kernel_size:     A scale.
    :param channel_index:   A tensor, sized [] representing the channel index.
    :return:                A tensor, sized [], the sum of all variances.
                            and A tensor, sized [n, n], the transformation.
    '''
    reproduction = kernel_size * kernel_size
    channel_num = cov_matrix.get_shape().as_list()[0]/reproduction
    proto_ = []
    for i in range(reproduction):
        proto_.append(i*channel_num)
    proto = tf.Variable(proto_)
    proto = tf.reshape(proto, [reproduction, 1])
    proto = tf.add(proto, channel_index)
    proto = tf.cast(proto, tf.int32)

    trans = group_decorrelation(cov_matrix, proto)

    one_hot_mask = tf.sparse_to_dense(proto, [int(channel_num*reproduction)], 1.0)
    one_hot_mask = tf.reshape(one_hot_mask, [int(channel_num*reproduction), 1])

    after_trans = tf.matmul(tf.matmul(trans, cov_matrix), tf.transpose(trans))

    after_trans_diag = tf.diag_part(after_trans)
    after_trans_diag = tf.reshape(after_trans_diag, [1, int(channel_num*reproduction)])

    scores = tf.matmul(after_trans_diag, one_hot_mask)

    return scores, trans


# PART 3: Compute scores for each activation and mask operations.
def find_min_value_and_index(the_list, the_mask):
    min_value = INF
    min_index = None
    for i in range(len(the_list)):
        if the_mask[i] == 1:
            value = the_list[i]
            if value < min_value:
                min_value = value
                min_index = i
    return min_value, min_index


class score_update_flow_builder(object):
    def __init__(self):
        self.weights_list = []
        self.weights_source_list = []
        self.mask_list = []
        self.grad_var_wise_list = None
        self.variance_after_de_correlation_list = []
        self.score_list = None
        self.stacked_trans_list = []
        self.sum_mask_list = []
        self.update_masks_for_zero_var_op_list = []

    def collect_weights_mask(self, weights_variable, mask, zero_var_mask_one):
        zero_var_mask_one = tf.cast(zero_var_mask_one, tf.float32)
        self.weights_source_list.append(weights_variable)
        shape = weights_variable.get_shape().as_list()
        reshaped = tf.transpose(tf.reshape(weights_variable, [shape[0]*shape[1]*shape[2], shape[3]]))
        self.weights_list.append(reshaped)
        self.update_masks_for_zero_var_op_list.append(tf.assign(mask, tf.multiply(mask, (1-zero_var_mask_one))))
        self.mask_list.append(mask)
        self.sum_mask_list.append(tf.reduce_sum(mask))

    def collect_variances_trans(self, variance, trans):
        self.variance_after_de_correlation_list.append(variance)
        self.stacked_trans_list.append(trans)

    def set_gard_var_wise_list(self, var_list):
        self.grad_var_wise_list = var_list

    def get_score_list(self):
        with tf.variable_scope('compute_score'):
            score_list = []
            if len(self.weights_list) == len(self.grad_var_wise_list):
                for i in range(len(self.weights_list)):
                    weights = self.weights_list[i]
                    # input_length = weights.get_shape().as_list()[0]
                    reform_matrix = self.grad_var_wise_list[i]
                    new_weights = tf.matmul(reform_matrix, weights)
                    score_scale = tf.norm(new_weights, axis=0)
                    score = tf.multiply(score_scale, self.variance_after_de_correlation_list[i])
                    score_list.append(score)
                    self.score_list = score_list
                    print('Build score flow.')
            else:
                print('ERROR: Length of weights_list and grad_var_wise_list are not equal ')
        return score_list

    def remove_last_and_update(self, sess):
        """
        Here we run sess and get masks, weights, scores and transformation.
        Then we update weights and masks.
        """
        if len(self.score_list) == 0:
            print('ERROR: Don\'t have scores.')
        else:
            sess.run(self.update_masks_for_zero_var_op_list)
            py_score_list = sess.run(self.score_list)
            py_masks_list = sess.run(self.mask_list)

            all_min = np.inf
            all_layer_index = None
            all_in_layer_index = None
            for layer_index in range(len(py_score_list)):
                layer_score = py_score_list[layer_index]
                min_score, index = find_min_value_and_index(layer_score, py_masks_list[layer_index])
                if min_score < all_min:
                    all_min = min_score
                    all_layer_index = layer_index
                    all_in_layer_index = index

            mask_to_update = py_masks_list[all_layer_index]
            mask_to_update[all_in_layer_index] = 0
            the_mask_tensor = self.mask_list[all_layer_index]
            sess.run(the_mask_tensor.assign(mask_to_update))

            py_trans = sess.run(self.stacked_trans_list[all_layer_index])
            transformation = py_trans[all_in_layer_index, :]

            py_weights = sess.run(self.weights_list[all_layer_index])

            for i in range(len(transformation)):
                py_weights[:, i] = py_weights[:, i] - transformation[i] * py_weights[:, all_in_layer_index]

            sess.run(self.weights_list[all_layer_index].assign(py_weights))

            """
            # This is a flow version. Not finished yet.
            
            min_list = []
            min_index_list = []
            length = len(self.score_list)
            for score in self.score_list:
                min_list.append(tf.reduce_min(score))
                min_index_list.append(tf.argmin(score))
            min_from_each_layer = tf.stack(min_list)
            the_layer_to_prune_on = tf.argmin(min_from_each_layer)
            min_indexes = tf.reshape(tf.stack(min_index_list), [length, 1])
            mask_to_choose_index = tf.sparse_to_dense(the_layer_to_prune_on, [1, length], 1)
            activation_index = tf.matmul(mask_to_choose_index, min_indexes)
            """
            return mask_to_update, py_weights

    def compute_score_and_remove_last(self, sess, last_num=1):
        # sess.run(self.update_masks_for_zero_var_op_list)
        # need to update parameters in bn
        print('Update mask for zero variance...')
        weights_list = sess.run(self.weights_list)
        print('Get weights...')
        reform_trans = sess.run(self.grad_var_wise_list)
        print('Get trans...')
        variances_list = sess.run(self.variance_after_de_correlation_list)
        print(variances_list)
        print('Get variances...')
        mask_list = sess.run(self.mask_list)
        print('Get masks...')
        all_min = np.inf
        all_layer_index = None
        all_in_layer_index = None
        for i in range(len(weights_list)):
            weight = weights_list[i]
            trans = reform_trans[i]
            new_weights = np.matmul(trans, weight)
            score_scale = np.linalg.norm(new_weights, 2, 0)
            score = np.multiply(score_scale, variances_list[i])
            mask = mask_list[i]
            min_score, index = find_min_value_and_index(score, mask)
            if min_score < all_min:
                all_min = min_score
                all_layer_index = i
                all_in_layer_index = index

        mask_to_update = mask_list[all_layer_index]
        mask_to_update[all_in_layer_index] = 0
        print('Prune the %dth layer, the %dth channel.' % (all_layer_index, all_in_layer_index))
        the_mask_tensor = self.mask_list[all_layer_index]
        sess.run(the_mask_tensor.assign(mask_to_update))
        print('Assign mask...')

        py_trans = sess.run(self.stacked_trans_list[all_layer_index])
        transformation = py_trans[all_in_layer_index, :]

        py_weights = sess.run(self.weights_list[all_layer_index])

        for i in range(len(transformation)):
            py_weights[:, i] = py_weights[:, i] - transformation[i] * py_weights[:, all_in_layer_index]

        py_weights = np.transpose(py_weights)
        py_weights = np.reshape(py_weights, [1, 1, py_weights.shape[0], py_weights.shape[1]])

        sess.run(self.weights_source_list[all_layer_index].assign(py_weights))
        print('Assign weights...')

        return all_layer_index, all_in_layer_index

    def compute_score_and_save(self, sess):
        sess.run(self.update_masks_for_zero_var_op_list)
        print('Update mask for zero variance...')
        weights_list = sess.run(self.weights_list)
        print('Get weights...')
        reform_trans = sess.run(self.grad_var_wise_list)
        print('Get trans...')
        variances_list = sess.run(self.variance_after_de_correlation_list)
        print('Get variances...')
        mask_list = sess.run(self.mask_list)
        print('Get masks...')
        with open('./score.txt', 'a') as file:
            for i in range(len(weights_list)):
                weight = weights_list[i]
                trans = reform_trans[i]
                new_weights = np.matmul(trans, weight)
                score_scale = np.linalg.norm(new_weights, 2, 0)
                score = np.multiply(score_scale, variances_list[i])
                mask = mask_list[i]
                result = score*mask
                file.write(result.tostring())

    def remaining_activation(self, sess):
        nums = sess.run(self.sum_mask_list)
        return sum(nums)


if __name__ == '__main__':
    a = tf.Variable([[1,0, 0.7], [0, 0, 0], [0.7, 0, 1.3]], dtype=tf.float32)

    show1, show2 = variances_after_decorrelation_and_corresponding_transformations(a,
                                                                                   tf.Variable([0,0.1,0], dtype=tf.float32),
                                                                                   tf.Variable([1,0,1], dtype=tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run([show1,show2]))