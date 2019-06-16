import tensorflow as tf
import numpy as np
# X = [1,16,16,8]
# x = tf.placeholder(name='x', dtype=tf.float32, shape=X)
# # print(x.shape)
# # y = tf.reduce_sum(x)
# conv1 = tf.contrib.layers.conv2d_transpose(inputs=x,num_outputs = 1, kernel_size=[2,2 ],stride = 2,padding = 'VALID',activation_fn  =tf.nn.relu)#(32,32) -> (28,28,8)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())#inputs,
#     u = sess.run([conv1],feed_dict = {x:np.random.random(X)})
#     print(np.array(u).shape)
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

# Create the op
inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init_op)
  inc_v1.op.run()
  dec_v2.op.run()

  # Save the variables to disk.
  save_path = saver.save(sess, "./dummymodel.ckpt")
