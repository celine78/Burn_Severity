from __future__ import absolute_import

import tensorflow as tf
from keras_core.layers import Layer, Dense, Embedding


class patch_extract(Layer):
    '''
    Extract patches from the input feature map.

    patches = patch_extract(patch_size)(feature_map)

    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner,
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020.
    An image is worth 16x16 words: Transformers for image recognition at scale.
    arXiv preprint arXiv:2010.11929.

    Input
    ----------
        feature_map: a four-dimensional tensor of (num_sample, width, height, channel)
        patch_size: size of split patches (width=height)

    Output
    ----------
        patches: a two-dimensional tensor of (num_sample*num_patch, patch_size*patch_size)
                 where `num_patch = (width // patch_size) * (height // patch_size)`

    For further information see: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches

    '''

    def __init__(self, patch_size, **kwargs):
        super(patch_extract, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]

    def call(self, images):
        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(images=images,
                                  sizes=(1, self.patch_size_x, self.patch_size_y, 1),
                                  strides=(1, self.patch_size_x, self.patch_size_y, 1),
                                  rates=(1, 1, 1, 1), padding='VALID', )
        # patches.shape = (num_sample, patch_num, patch_num, patch_size*channel)

        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        patches = tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))
        # patches.shape = (num_sample, patch_num*patch_num, patch_size*channel)

        return patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({'patch_size': self.patch_size, })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class patch_embedding(Layer):
    '''
    Embed patches to tokens.

    patches_embed = patch_embedding(num_patch, embed_dim)(pathes)

    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner,
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020.
    An image is worth 16x16 words: Transformers for image recognition at scale.
    arXiv preprint arXiv:2010.11929.

    Input
    ----------
        num_patch: number of patches to be embedded.
        embed_dim: number of embedded dimensions.

    Output
    ----------
        embed: Embedded patches.

    For further information see: https://keras.io/api/layers/core_layers/embedding/

    '''

    def __init__(self, num_patch, embed_dim, **kwargs):
        super(patch_embedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.proj = Dense(embed_dim)
        self.pos_embed = Embedding(input_dim=num_patch, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        embed = self.proj(patch) + self.pos_embed(pos)
        return embed
