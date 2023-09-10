# TransU-Net++: Rethinking Attention Gated TransU-Net for Deforestation Mapping
# Ali Jamali, Swalpa Kumar Roy, Jonathan Li, and Pedram Ghamisi
# https://github.com/aj1365/TransUNetplus2




from __future__ import absolute_import

import keras_core.src.activations
import torch
import torch.nn as nn

from unet.utils.keras_core_unet_collection.layer_utils import decode_layer, CONV_output
from unet.utils.keras_core_unet_collection._model_unet_2d import UNET_left, UNET_right
from unet.utils.keras_core_unet_collection.transformer_layers import patch_extract, patch_embedding
from unet.utils.keras_core_unet_collection._backbone_zoo import backbone_zoo

import tensorflow as tf
from keras_core.optimizers import Adam
from keras_core.layers import Input, Add, Conv2D, MultiHeadAttention, LayerNormalization, Dense, BatchNormalization, \
    MaxPooling2D, AveragePooling2D, Activation
from keras_core.models import Model


def HetConv(feature_map, conv_filter, groups):
    # Groupwise Convolution
    x1 = Conv2D(filters=conv_filter, kernel_size=(3, 3), groups=groups, padding='same')(feature_map)

    # Pointwise Convolution
    x2 = Conv2D(filters=conv_filter, kernel_size=(1, 1), strides=1, padding='same')(feature_map)

    addition = Add()([x1, x2])

    return addition

def CONV_stack(X, channel, kernel_size=3, stack_num=2,
               dilation_rate=1, activation='ReLU',
               batch_norm=False, name='conv_stack'):
    bias_flag = not batch_norm

    # stacking Convolutional layers
    for i in range(stack_num):

        X = HetConv(X, conv_filter=channel, groups=4)

        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_{}_bn'.format(name, i))(X)

        # activation
        #activation_func = eval(activation)
        activation_func = keras_core.layers.ReLU
        X = activation_func(name='{}_{}_activation'.format(name, i))(X)

    return X


def encode_layer(X, channel, pool_size, pool, kernel_size='auto',
                 activation='ReLU', batch_norm=False, name='encode'):
    # parsers
    if (pool in [False, True, 'max', 'ave']) is not True:
        raise ValueError('Invalid pool keyword')

    # maxpooling2d as default
    if pool is True:
        pool = 'max'

    elif pool is False:
        # stride conv configurations
        bias_flag = not batch_norm

    if pool == 'max':
        X = MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_maxpool'.format(name))(X)

    elif pool == 'ave':
        X = AveragePooling2D(pool_size=(pool_size, pool_size), name='{}_avepool'.format(name))(X)

    else:
        if kernel_size == 'auto':
            kernel_size = pool_size

        # linear convolution with strides
        X = Conv2D(channel, kernel_size, strides=(pool_size, pool_size),
                   padding='valid', use_bias=bias_flag, name='{}_stride_conv'.format(name))(X)

        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_bn'.format(name))(X)

        # activation
        if activation is not None:
            #activation_func = eval(activation)
            activation_func = keras_core.layers.ReLU
            X = activation_func(name='{}_activation'.format(name))(X)

    return X


def attention_gate(X, g, channel,
                   activation='ReLU',
                   attention='add', name='att'):
    # activation_func = eval(activation)
    activation_func = keras_core.layers.ReLU
    # attention_func = eval(attention)
    attention_func = keras_core.layers.Attention('add')

    # mapping the input tensor to the intermediate channel
    theta_att = Conv2D(channel, 1, use_bias=True, name='{}_theta_x'.format(name))(X)

    # mapping the gate tensor
    phi_g = Conv2D(channel, 1, use_bias=True, name='{}_phi_g'.format(name))(g)

    # ----- attention learning ----- #
    query = attention_func([theta_att, phi_g], name='{}_add'.format(name))

    # nonlinear activation
    f = activation_func(name='{}_activation'.format(name))(query)

    # linear transformation
    psi_f = Conv2D(1, 1, use_bias=True, name='{}_psi_f'.format(name))(f)
    # ------------------------------ #

    # sigmoid activation as attention coefficients
    coef_att = Activation('sigmoid', name='{}_sigmoid'.format(name))(psi_f)

    # multiplicative attention masking
    X_att = tf.keras.layers.multiply([X, coef_att], name='{}_masking'.format(name))

    return X_att


def UNET_left(X, channel, kernel_size=3, stack_num=2, activation='ReLU',
              pool=True, batch_norm=False, name='left0'):
    pool_size = 2

    X = encode_layer(X, channel, pool_size, pool, activation=activation,
                     batch_norm=batch_norm, name='{}_encode'.format(name))

    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm, name='{}_conv'.format(name))

    return X


def UNET_att_right(X, X_left, channel, att_channel, kernel_size=3, stack_num=2,
                   activation='ReLU', atten_activation='ReLU', attention='add',
                   unpool=True, batch_norm=False, name='right0'):
    X = HetConv(X, conv_filter=channel, groups=4)

    pool_size = 2

    X = decode_layer(X, channel, pool_size, unpool,
                     activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))

    X_left = attention_gate(X=X_left, g=X, channel=att_channel, activation=atten_activation,
                            attention=attention, name='{}_att'.format(name))

    # Tensor concatenation
    H = tf.keras.layers.concatenate([X, X_left], axis=-1, name='{}_concat'.format(name))

    # stacked linear convolutional layers after concatenation
    H = CONV_stack(H, channel, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm, name='{}_conv_after_concat'.format(name))

    return H


def ViT_MLP(X, filter_num, activation='GELU', name='MLP'):
    #activation_func = eval(activation)
    activation_func = keras_core.layers.Activation('gelu')

    for i, f in enumerate(filter_num):
        X = Dense(f, name='{}_dense_{}'.format(name, i))(X)
        X = activation_func(X)

    return X


def ViT_block(V, num_heads, key_dim, filter_num_MLP, activation='GELU', name='ViT'):
    # Multiheaded self-attention (MSA)
    V_atten = V  # <--- skip
    V_atten = LayerNormalization(name='{}_layer_norm_1'.format(name))(V_atten)
    V_atten = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                 name='{}_atten'.format(name))(V_atten, V_atten)
    # Skip connection
    V_add = tf.keras.layers.add([V_atten, V], name='{}_skip_1'.format(name))  # <--- skip

    # MLP
    V_MLP = V_add  # <--- skip
    V_MLP = LayerNormalization(name='{}_layer_norm_2'.format(name))(V_MLP)
    V_MLP = ViT_MLP(V_MLP, filter_num_MLP, activation, name='{}_mlp'.format(name))
    # Skip connection
    V_out = tf.keras.layers.add([V_MLP, V_add, V], name='{}_skip_2'.format(name))  # <--- skip

    return V_out


def transunet_plus2_base(input_tensor: torch.Tensor, filter_num, stack_num_down=2, stack_num_up=2,
                         embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                         activation='ReLU', atten_activation='ReLU', attention='add', mlp_activation='GELU',
                         batch_norm=False, pool=True, unpool=True,
                         backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True,
                         name='transunet_plus2'):

    X_skip = []
    depth_ = len(filter_num)

    # ----- internal parameters ----- #

    # patch size (fixed to 1-by-1)
    patch_size = 1

    # input tensor size
    input_size = input_tensor.shape[1]

    # encoded feature map size
    encode_size = input_size // 2 ** (depth_ - 1)

    # number of size-1 patches
    num_patches = encode_size ** 2

    # dimension of the attention key (= dimension of embedings)
    key_dim = embed_dim

    # number of MLP nodes
    filter_num_MLP = [num_mlp, embed_dim]

    # ----- UNet-like downsampling ----- #

    # no backbone cases
    if backbone is None:

        X = input_tensor

        X = HetConv(X, conv_filter=filter_num[0], groups=8)
        X_skip.append(X)

        # downsampling blocks
        for i, f in enumerate(filter_num[1:]):
            X = UNET_left(X, f, stack_num=stack_num_down, activation=activation, pool=pool,
                          batch_norm=batch_norm, name='{}_down{}'.format(name, i + 1))
            X_skip.append(X)

    # backbone cases
    else:
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_skip = backbone_([input_tensor, ])
            depth_encode = len(X_skip)

        # for other backbones
        else:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_ - 1, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_skip = backbone_([input_tensor, ])
            depth_encode = len(X_skip) + 1

        # extra conv2d blocks are applied
        # if downsampling levels of a backbone < user-specified downsampling levels
        if depth_encode < depth_:

            # begins at the deepest available tensor
            X = X_skip[-1]

            # extra downsamplings
            for i in range(depth_ - depth_encode):
                i_real = i + depth_encode

                X = UNET_left(X, filter_num[i_real], stack_num=stack_num_down, activation=activation, pool=pool,
                              batch_norm=batch_norm, name='{}_down{}'.format(name, i_real + 1))
                X_skip.append(X)

    # subtrack the last tensor (will be replaced by the ViT output)
    X = X_skip[-1]
    X_skip = X_skip[:-1]

    # 1-by-1 linear transformation before entering ViT blocks

    X = HetConv(X, conv_filter=filter_num[-1], groups=4)

    X = patch_extract((patch_size, patch_size))(X)
    X = patch_embedding(num_patches, embed_dim)(X)

    # stacked ViTs
    for i in range(num_transformer):
        X = ViT_block(X, num_heads, key_dim, filter_num_MLP, activation=mlp_activation,
                      name='{}_ViT_{}'.format(name, i))

    # reshape patches to feature maps
    X = tf.convert_to_tensor(X)
    print(type(X))
    X = tf.reshape(X, (-1, encode_size, encode_size, embed_dim))

    X = HetConv(X, conv_filter=filter_num[-1], groups=4)

    X_skip.append(X)

    # ----- UNet-like upsampling ----- #

    # reverse indexing encoded feature maps
    X_skip = X_skip[::-1]
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)

    # reverse indexing filter numbers
    filter_num_decode = filter_num[:-1][::-1]

    # upsampling with concatenation
    for i in range(depth_decode):
        f = filter_num_decode[i]

        X = UNET_att_right(X, X_decode[i], f, att_channel=f // 2, stack_num=stack_num_up,
                           activation=activation, atten_activation=atten_activation, attention=attention,
                           unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i))

    # if tensors for concatenation is not enough
    # then use upsampling without concatenation
    if depth_decode < depth_ - 1:
        for i in range(depth_ - depth_decode - 1):
            i_real = i + depth_decode
            X = UNET_right(X, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation,
                           unpool=unpool, batch_norm=batch_norm, concat=False, name='{}_up{}'.format(name, i_real))

    return X


def transunet_plus2(input_size, filter_num, n_labels, stack_num_down=1, stack_num_up=1,
                    embed_dim=44, num_mlp=252, num_heads=4, num_transformer=1,
                    activation='ReLU', atten_activation='ReLU', attention='add', mlp_activation='GELU',
                    output_activation='Sigmoid', batch_norm=False, pool=True, unpool=True,
                    backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True,
                    name='transunet_plus2'):

    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    f1 = tf.keras.metrics.F1Score(name='f1', average='micro', threshold=0.4)
    sgd_optimizer = Adam()

    IN = Input(input_size)

    # base
    X = transunet_plus2_base(IN, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                             embed_dim=embed_dim, num_mlp=num_mlp, num_heads=num_heads, num_transformer=num_transformer,
                             activation=activation, atten_activation=atten_activation, attention=attention,
                             mlp_activation=mlp_activation, batch_norm=batch_norm, pool=pool, unpool=unpool,
                             backbone=backbone, weights=weights, freeze_backbone=freeze_backbone,
                             freeze_batch_norm=freeze_batch_norm, name=name)

    # output layer
    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))

    # functional API model
    model = Model(inputs=[IN, ], outputs=[OUT, ], name='{}_model'.format(name))
    model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1])
    #

    return model


if __name__ == '__main__':
    model = transunet_plus2(input_size=(512, 512, 8), filter_num=[16, 32, 64, 128], n_labels=2)
