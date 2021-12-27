# Here the definition of the WGAN-GP class with the network specificities
 
import tensorflow as tf

class WGAN(tf.keras.Model):
    """[summary]
    I used github/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/ as a reference on this.
    
    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(WGAN, self).__init__()
        self.__dict__.update(kwargs)

        self.gen = tf.keras.Sequential(self.gen)
        self.disc = tf.keras.Sequential(self.disc)
        
    def generate(self, z):
        return self.gen(z)

    def discriminate(self, x):
        return self.disc(x)

    def compute_loss(self, x):
        """ passes through the network and computes loss
        """
        ### pass through network
        # generating noise from a uniform distribution
        z_samp = tf.random.normal([x.shape[0],2, self.n_Z])
        
        # run noise through generator
        x_gen = self.generate(z_samp)

        # discriminate x and x_gen
        logits_x = self.discriminate(x)
        logits_x_gen = self.discriminate(x_gen)

        # gradient penalty
        d_regularizer = self.gradient_penalty(x, x_gen)
        
        ### aggiungere la wassertein loss tra x_rna_gen e x_atac_gen
        ### losses
        disc_loss1 = (
            tf.reduce_mean(logits_x)
            - tf.reduce_mean(logits_x_gen)
            + d_regularizer * self.gradient_penalty_weight
        )

        # losses of fake with label "1"
        gen_loss = tf.reduce_mean(logits_x_gen)  
                   
        return disc_loss1, gen_loss

    def compute_gradients(self, x):
        """ passes through the network and computes loss
        """
        ### pass through network
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape1: 
            disc_loss1, gen_loss = self.compute_loss(x)

        # compute gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.gen.trainable_variables)

        disc_gradients1 = disc_tape1.gradient(disc_loss1, self.disc.trainable_variables)

        return gen_gradients, disc_gradients1 

    def apply_gradients(self, gen_gradients, disc_gradients1): 

        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.gen.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients1, self.disc.trainable_variables)
        )

    def gradient_penalty(self, x, x_gen):
        
        epsilon = tf.random.uniform([x.shape[0],1, 1], 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.discriminate(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer


    @tf.function
    def train(self, train_x):
        gen_gradients, disc_gradients1 = self.compute_gradients(train_x)
        self.apply_gradients(gen_gradients, disc_gradients1)
        
def get_model():
      return WGAN(
    gen = generator,
    disc = discriminator,
    gen_optimizer = gen_optimizer,
    disc_optimizer = disc_optimizer,
    n_Z = N_Z,
    gradient_penalty_weight = 10.0,
              name='WGAN')
              
N_Z = 1024

generator = [

    tf.keras.layers.Conv1D(filters=512, kernel_size=2, strides=1, padding='same', activation="relu"),    
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(filters=124, kernel_size=2, strides=1, padding='same', activation="relu"),    
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(filters=5, kernel_size=2, strides=1, padding='same', activation="relu")
]

discriminator = [
    
    tf.keras.layers.InputLayer(input_shape=(2,5)),
    tf.keras.layers.Conv1D(filters=124, kernel_size=2, strides=1, padding='same', activation="relu"),
    tf.keras.layers.Conv1D(filters=512, kernel_size=2, strides=1, padding='same', activation="relu"),
    tf.keras.layers.Dense(units=1)
]                               

# optimizers
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.9, epsilon=1e-07, amsgrad=False)
disc_optimizer = tf.keras.optimizers.RMSprop(0.0005)