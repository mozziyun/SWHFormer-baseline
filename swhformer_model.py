
import tensorflow as tf
from tensorflow.keras import layers as L, Model
from tensorflow.keras import backend as K

# ---------------- metrics ----------------
def corr_metric(y_true, y_pred):
    x = K.flatten(y_true)
    y = K.flatten(y_pred)
    xm, ym = x - K.mean(x), y - K.mean(y)
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(xm * xm) * K.sum(ym * ym)) + 1e-12
    return r_num / r_den

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# ---------------- 1) seq -> 3ch ----------------
def seq_to_3ch(x, smooth=True):
    if smooth:
        x = L.AveragePooling3D(pool_size=(1,3,3), strides=(1,1,1), padding='same')(x)
    mean = tf.reduce_mean(x, axis=1)
    std  = tf.math.reduce_std(x, axis=1)
    diff = tf.reduce_mean(tf.abs(x[:,1:] - x[:,:-1]), axis=1)
    return tf.concat([mean, std, diff], axis=-1)  # (B,H,W,3)

# ---------------- 2) Patch Embedding ----------------
class PatchEmbed(L.Layer):
    def __init__(self, embed_dim=768, patch=16):
        super().__init__()
        self.patch = patch
        self.proj = L.Conv2D(embed_dim, kernel_size=patch, strides=patch, padding='valid')

    def call(self, x):
        x = self.proj(x)
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]
        return tf.reshape(x, [B, H * W, C])

# ---------------- 3) CLS + Pos ----------------
class AddClassPos(L.Layer):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.cls = self.add_weight(
            name="cls", shape=(1, 1, embed_dim), initializer="zeros", trainable=True
        )
        self.pos = self.add_weight(
            name="pos", shape=(1, seq_len, embed_dim),
            initializer="random_normal", trainable=True
        )

    def call(self, tokens):
        B = tf.shape(tokens)[0]
        cls = tf.tile(self.cls, [B, 1, 1])
        z = tf.concat([cls, tokens], axis=1)
        return z + self.pos[:, :tf.shape(z)[1], :]

# ---------------- 4) Transformer block ----------------
def encoder_block(x, num_heads=12, mlp_ratio=4.0, drop=0.0):
    C = x.shape[-1]
    y = L.LayerNormalization(epsilon=1e-6)(x)
    y = L.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=C // num_heads,
        dropout=drop
    )(y, y)
    x = x + y

    y = L.LayerNormalization(epsilon=1e-6)(x)
    y = L.Dense(int(C * mlp_ratio), activation=tf.keras.activations.gelu)(y)
    y = L.Dropout(drop)(y)
    y = L.Dense(C)(y)
    return x + y

# ---------------- 5) build model (wind 제거 버전) ----------------
def build_swhformer_no_wind(input_shape=(64, 128, 128, 1),
                            img_size=384, embed_dim=768, depth=12, heads=12,
                            mlp_ratio=4.0, drop=0.0, patch=32):
    if img_size % patch != 0:
        raise ValueError(f"img_size({img_size}) must be divisible by patch({patch}).")

    inp_img = L.Input(shape=input_shape, name="radar_seq")
    x = tf.cast(inp_img, tf.float32)
    x = seq_to_3ch(x, smooth=True)                 # (B,128,128,3)
    x = tf.image.resize(x, (img_size, img_size))   # (B,img_size,img_size,3)

    tokens = PatchEmbed(embed_dim=embed_dim, patch=patch)(x)
    n_patches = (img_size // patch) * (img_size // patch)

    z = AddClassPos(seq_len=1 + n_patches, embed_dim=embed_dim)(tokens)

    for _ in range(depth):
        z = encoder_block(z, num_heads=heads, mlp_ratio=mlp_ratio, drop=drop)

    z = L.LayerNormalization(epsilon=1e-6)(z)
    cls = L.Lambda(lambda x: x[:,0])(z)
    h = L.Dense(64, activation='swish')(cls)
    out = L.Dense(1, activation='linear', name="swh")(h)

    return Model(inp_img, out, name=f"SWHFormer_NoWind_p{patch}")


if __name__ == "__main__":
    input_shape = (64, 128, 128, 1)

    model = build_swhformer_no_wind(input_shape=input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='mse',
        metrics=['mae', rmse, corr_metric]
    )

    model.summary()
