#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import time
from ctypes import *

caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

#输入参数-------------->
validate_prototxt = ''
validate_caffemodel = ''
validate_maxitem = 500

def main(args):
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(0)

    #创建网络
    net = caffe.Net(validate_prototxt,validate_caffemodel,caffe.TEST)

    #前向运算
    for i in range(validate_maxitem):
        print("----------------------net.forward -------------------->[%d]" % i)
        output = net.forward()
        #time.sleep( 1 )
    print("------------------------AccuracyIOU=%f------------------------" % output['accuracy_iou'])    
    #python能够改变变量作用域的代码段是def、class、lamda.
    #if/elif/else、try/except/finally、for/while 并不能涉及变量作用域的更改，
    #也就是说他们的代码块中的变量，在外部也是可以访问的

if __name__ == '__main__':
    if len(sys.argv[1:]) < 2:
        print("usage: .py validate.prototxt validate.caffemodel")
        exit()
    validate_prototxt = sys.argv[1]
    validate_caffemodel = sys.argv[2]
    main(None)
    pass