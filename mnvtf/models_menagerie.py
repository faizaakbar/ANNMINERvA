def make_menndl_633167(img_depth=2):
    """
    epsilon topology compatible; assume NHWC data format exclusively
    """
    convpooldict = {}
    convpooldict['use_batch_norm'] = False
    conv_padding_scheme = 'VALID'
    pool_padding_scheme = 'VALID'

    # build 3 convolutional towers. turn off pooling at any given
    # layer by setting `convpooldict_view['poolLayerNumber'] = None`

    # build the '_x' tower
    convpooldict_x = {}
    convpooldict_x['conv1'] = {}    # conv1_0
    convpooldict_x['pool1'] = None  # pool2_0 (identity)
    convpooldict_x['conv2'] = {}    # conv3_0
    convpooldict_x['pool2'] = None
    convpooldict_x['conv3'] = {}    # conv4_0
    convpooldict_x['pool3'] = {}    # pool5_0
    # conv1_0
    h, w, fin, fout = 8, 3, img_depth, 32
    c1_stride_h, c1_stride_w = 2, 1
    convpooldict_x['conv1']['kernels'] = [h, w, fin, fout]
    convpooldict_x['conv1']['biases'] = [32]
    convpooldict_x['conv1']['strides'] = [1, c1_stride_h, c1_stride_w, 1]
    convpooldict_x['conv1']['apply_dropout'] = True
    convpooldict_x['conv1']['padding'] = conv_padding_scheme
    # pool2_0 was an identity mapping
    # conv3_0
    h, w, fin, fout = 6, 7, 32, 20
    convpooldict_x['conv2']['kernels'] = [h, w, fin, fout]
    convpooldict_x['conv2']['biases'] = [20]
    convpooldict_x['conv2']['strides'] = [1, 1, 1, 1]
    convpooldict_x['conv2']['apply_dropout'] = False
    convpooldict_x['conv2']['padding'] = conv_padding_scheme
    # conv4_0
    h, w, fin, fout = 12, 39, 20, 28
    convpooldict_x['conv3']['kernels'] = [h, w, fin, fout]
    convpooldict_x['conv3']['biases'] = [28]
    convpooldict_x['conv3']['strides'] = [1, 1, 1, 1]
    convpooldict_x['conv3']['apply_dropout'] = False
    convpooldict_x['conv3']['padding'] = conv_padding_scheme
    # pool5_0
    convpooldict_x['pool3']['ksize'] = [1, 2, 1, 1]
    convpooldict_x['pool3']['strides'] = [1, 2, 1, 1]
    convpooldict_x['pool3']['padding'] = pool_padding_scheme
    convpooldict_x['n_layers'] = 3
    # at the end of the tower, flatten and possibly add dense layers...
    convpooldict['x'] = convpooldict_x

    # build the '_u' tower
    convpooldict_u = {}
    convpooldict_u['conv1'] = {}    # conv1_1
    convpooldict_u['pool1'] = None  # pool2_1 (identity)
    convpooldict_u['conv2'] = {}    # conv3_1
    convpooldict_u['pool2'] = None
    convpooldict_u['conv3'] = {}    # conv4_1
    convpooldict_u['pool3'] = {}    # pool5_1
    # conv1_1
    h, w, fin, fout = 8, 3, img_depth, 32
    c1_stride_h, c1_stride_w = 2, 1
    convpooldict_u['conv1']['kernels'] = [h, w, fin, fout]
    convpooldict_u['conv1']['biases'] = [32]
    convpooldict_u['conv1']['strides'] = [1, c1_stride_h, c1_stride_w, 1]
    convpooldict_u['conv1']['apply_dropout'] = True
    convpooldict_u['conv1']['padding'] = conv_padding_scheme
    # pool2_1 was an identity mapping
    # conv3_1
    h, w, fin, fout = 6, 7, 32, 20
    convpooldict_u['conv2']['kernels'] = [h, w, fin, fout]
    convpooldict_u['conv2']['biases'] = [20]
    convpooldict_u['conv2']['strides'] = [1, 1, 1, 1]
    convpooldict_u['conv2']['apply_dropout'] = False
    convpooldict_u['conv2']['padding'] = conv_padding_scheme
    # conv4_1
    h, w, fin, fout = 12, 39, 20, 28
    convpooldict_u['conv3']['kernels'] = [h, w, fin, fout]
    convpooldict_u['conv3']['biases'] = [28]
    convpooldict_u['conv3']['strides'] = [1, 1, 1, 1]
    convpooldict_u['conv3']['apply_dropout'] = False
    convpooldict_u['conv3']['padding'] = conv_padding_scheme
    # pool5_1
    convpooldict_u['pool3']['ksize'] = [1, 2, 1, 1]
    convpooldict_u['pool3']['strides'] = [1, 2, 1, 1]
    convpooldict_u['pool3']['padding'] = pool_padding_scheme
    convpooldict_u['n_layers'] = 3
    # at the end of the tower, flatten and possibly add dense layers...
    convpooldict['u'] = convpooldict_u

    # build the '_v' tower
    convpooldict_v = {}
    convpooldict_v['conv1'] = {}    # conv1_2
    convpooldict_v['pool1'] = None  # pool2_2 (identity)
    convpooldict_v['conv2'] = {}    # conv3_2
    convpooldict_v['pool2'] = None
    convpooldict_v['conv3'] = {}    # conv4_2
    convpooldict_v['pool3'] = {}    # pool5_2
    # conv1_2
    h, w, fin, fout = 8, 3, img_depth, 32
    c1_stride_h, c1_stride_w = 2, 1
    convpooldict_v['conv1']['kernels'] = [h, w, fin, fout]
    convpooldict_v['conv1']['biases'] = [32]
    convpooldict_v['conv1']['strides'] = [1, c1_stride_h, c1_stride_w, 1]
    convpooldict_v['conv1']['apply_dropout'] = True
    convpooldict_v['conv1']['padding'] = conv_padding_scheme
    # pool2_2 was an identity mapping
    # conv3_2
    h, w, fin, fout = 6, 7, 32, 20
    convpooldict_v['conv2']['kernels'] = [h, w, fin, fout]
    convpooldict_v['conv2']['biases'] = [20]
    convpooldict_v['conv2']['strides'] = [1, 1, 1, 1]
    convpooldict_v['conv2']['apply_dropout'] = False
    convpooldict_v['conv2']['padding'] = conv_padding_scheme
    # conv4_2
    h, w, fin, fout = 12, 39, 20, 28
    convpooldict_v['conv3']['kernels'] = [h, w, fin, fout]
    convpooldict_v['conv3']['biases'] = [28]
    convpooldict_v['conv3']['strides'] = [1, 1, 1, 1]
    convpooldict_v['conv3']['apply_dropout'] = False
    convpooldict_v['conv3']['padding'] = conv_padding_scheme
    # pool5_2
    convpooldict_v['pool3']['ksize'] = [1, 2, 1, 1]
    convpooldict_v['pool3']['strides'] = [1, 2, 1, 1]
    convpooldict_v['pool3']['padding'] = pool_padding_scheme
    convpooldict_v['n_layers'] = 3
    # at the end of the tower, flatten and possibly add dense layers...
    convpooldict['v'] = convpooldict_v

    # at the end of all three towers, concatenate the output

    convpooldict['final_mlp'] = {}
    convpooldict['final_mlp']['n_dense_layers'] = 2
    convpooldict['final_mlp']['dense_n_output1'] = 64  # ip6_0
    convpooldict['final_mlp']['dense_n_output2'] = 64  # ip7_0
    # finalip == logits layer

    convpooldict['regularizer'] = {}
    convpooldict['regularizer']['type'] = None
    convpooldict['regularizer']['scale'] = 0.0001

    convpooldict['use_batch_norm'] = False

    return convpooldict
