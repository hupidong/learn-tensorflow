import tensorflow as  tf

config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)

with tf.device("/gpu:0"):
    rand_t=tf.random_uniform([50,50],0,10,dtype=tf.float32, seed=0)
    a=tf.Variable(rand_t)
    b=tf.Variable(rand_t)
    c=tf.matmul(a, b)
    init=tf.global_variables_initializer()
    
with tf.Session(config=config) as sess:
    sess.run(init)
    print(sess.run(c))
    
    
# 选择多个设备
c=[]
for d in ["/cpu:0","/cpu:1"]:
    with tf.device(d):
        rand_t=tf.random_uniform([50,50],0,10,dtype=tf.float32, seed=0)
        a=tf.Variable(rand_t)
        b=tf.Variable(rand_t)
        c.append(tf.matmul(a, b))
        init=tf.global_variables_initializer()
        
with tf.Session(config=config) as sess:
    sess.run(init)
    print(sess.run(c))