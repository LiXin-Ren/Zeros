import tensorflow as tf
a = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
b = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
queue = tf.train.slice_input_producer([a, b], shuffle=True, num_epochs=5)
queue1 = queue[0]
queue2 = queue[1]
x, y = tf.train.shuffle_batch([queue1, queue2], batch_size=10, capacity=200, min_after_dequeue=20)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 1
    try:
        while not coord.should_stop():
            print("i:", i)
            print("batch_x: ", sess.run(x))
            print("batch_y: ", sess.run(y))
           # print("queue: ", sess.run(queue))
            i += 1
    except tf.errors.OutOfRangeError:
        print("Done training -- epoch limit reached")