../build/tools/ristretto quantize -model ./captcha/captcha_deploy.prototxt -weights ./captcha/captcha_iter_50000.caffemodel -quantize_cfg ./captcha/bw_8.cfg -model_quantized ./captcha/qz/ -iterations 100 -error_margin 5 -gpu all
#-debug_out_float "./dumpout_base/" -debug_out_trim "./dumpout_trim/"
