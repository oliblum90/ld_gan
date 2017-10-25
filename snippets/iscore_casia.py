
sys.path.append("/export/home/oblum/projects/facenet/scr")
import facenet
import tensorflow as tf


model = face.Encoder()


#fname = "/export/home/oblum/projects/ld_gan/eval_imgs/iscores/face_net_model/20170512-110547/20170512-110547/20170512-110547.pb"

#tf.import_graph_def(fname)


# load network
#X = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
#net, end_points = inception_resnet_v1.inception_resnet_v1(X, is_training=False)

#saver = tf.train.Saver()
#sess = tf.Session()
#saver.restore(sess, fname)
