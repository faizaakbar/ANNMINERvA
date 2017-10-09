from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/tmp/data',
                           """Directory where data is stored.""")
tf.app.flags.DEFINE_string('file_root', 'mnv_data_',
                           """File basename.""")
tf.app.flags.DEFINE_boolean('do_something', True,
                            """Do a thing.""")
tf.app.flags.DEFINE_boolean('do_something_else', False,
                            """Do a different thing.""")


def main(argv=None):
    import glob
    print(FLAGS.data_dir.split(','))
    print(FLAGS.file_root.split(','))
    print(FLAGS.do_something)
    print(FLAGS.do_something_else)

    dirs = FLAGS.data_dir.split(',')
    fils = FLAGS.file_root.split(',')

    flist = []

    for d in dirs:
        for f in fils:
            flist.extend(
                glob.glob(d + '/' + f + '*')
            )

    for f in flist:
        print(f)


if __name__ == '__main__':
    tf.app.run()
