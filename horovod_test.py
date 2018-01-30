#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import horovod.tensorflow as hvd


def main(_):
    print("Initialize Horovod.")
    hvd.init()
    print("Initialize succeeded.")

if __name__ == "__main__":
    tf.app.run()
