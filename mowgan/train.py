"""
github/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/ was used as a reference for the WGAN-GP implementation
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.manifold import SpectralEmbedding
import anndata
import scanpy as sc
import pickle


class MOWGAN:
    def __init__(self, data, query, batch=None, mode="global",
                 n_dim=15, fill=[512,128], n_epochs=10000,
                 n_samples=5000, save_name=None, path=""):

        self.data = data                      # list of anndata
        self.query = query                    # list of embeddings 
        self.batch = batch                    # list of batch keys (only used in batch mode)
        self.mode = mode                      # "global" or "batch"
        self.n_dim = n_dim                    # number of features in the embedding
        self.fill = fill                      # list of filters
        self.n_epochs = n_epochs              # number of training epochs
        self.n_samples = n_samples            # number of samples in the generated
        self.save_name = save_name or []      # list of names from MOWGAN data
        self.path = path                      # path to the working directory
        self.N_Z = 1024
        self.model = None
        self.scalers = {}                     # store scalers per dataset/batch

    ################################
    # Preprocess
    ################################
    def preprocess_batches(self):
        for i, ad in enumerate(self.data):

            # ----- Batch relabeling (only if batch mode) -----
            if self.mode == "batch" and self.batch is not None:
                rename_batch = {}
                key = self.batch[0] if len(self.batch) == 1 else self.batch[i]
                cat = ad.obs[key].cat.categories
                for b in range(len(cat)):
                    rename_batch[cat[b]] = str(b)
                ad.obs['batch_train'] = ad.obs[key].map(rename_batch).astype('category')

            # ----- Dense + float32 -----
            if hasattr(ad.X, "todense"):
                ad.X = ad.X.toarray()

            ad.X = np.array(ad.X).astype('float32')
            ad.obsm[self.query[i]] = ad.obsm[self.query[i]].astype('float32')

    ################################
    # Dataset preparation
    ################################
    def prepare_dataset(self, batch_idx=None, BATCH_SIZE=256, TEST_SIZE=0.3):

        data_tr, r, d = {}, {}, {}

        for i, ad in enumerate(self.data):
            scaler = MinMaxScaler().fit(ad.obsm[self.query[i]][:, :self.n_dim])

            key = f"scaler{i}" if batch_idx is None else f"scaler{i}_{batch_idx}"
            self.scalers[key] = scaler
            data_tr[i] = scaler.transform(ad.obsm[self.query[i]][:, :self.n_dim])

            ad.obs['spectral_emb'] = SpectralEmbedding(
                n_components=1, affinity='precomputed'
            ).fit_transform(ad.obsp['connectivities'])

            sorted_idx = np.argsort(ad.obs['spectral_emb'])
            r[i] = data_tr[i][sorted_idx]

            d[i] = np.real(np.linalg.eig(
                sc._utils.get_igraph_from_adjacency(
                    ad.obsp['connectivities'][sorted_idx].todense(),
                    directed=False
                ).laplacian(normalized=True)
            )[1][:, 1])

        B = min([r[i].shape[0] for i in r])
        rIdx = np.sort(np.random.randint(0, B, size=BATCH_SIZE))
        train_mod = {0: r[0][rIdx]}

        reg = linear_model.BayesianRidge()
        reg.fit(r[0][rIdx], d[0][rIdx])

        for i in range(1, len(r)):
            scores, Idx = [], []
            for _ in range(50):
                idx = np.sort(np.random.randint(0, B, size=BATCH_SIZE))
                s = reg.score(r[i][idx], d[i][idx])
                Idx.append(idx)
                scores.append(s)
            best_idx = Idx[np.argmax(scores)]
            train_mod[i] = r[i][best_idx]

        c_train = np.concatenate(list(train_mod.values()), axis=1)
        real_x = np.reshape(c_train, (c_train.shape[0], len(self.data), self.n_dim))

        train, test = train_test_split(real_x, test_size=TEST_SIZE)

        train_ds = tf.data.Dataset.from_tensor_slices(train).batch(BATCH_SIZE)
        test_ds = tf.data.Dataset.from_tensor_slices(test).batch(BATCH_SIZE)

        # ---------- Automatically save scalers ----------
        self.save_scalers(batch_idx=batch_idx)

        return train_ds, test_ds

    ################################
    # Build WGAN-GP
    ################################
    def build_model(self):

        generator = [
            tf.keras.layers.Conv1D(512, len(self.data), padding='same', activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(128, len(self.data), padding='same', activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(self.n_dim, len(self.data), padding='same', activation="relu"),
        ]

        discriminator = [
            tf.keras.layers.InputLayer(shape=(len(self.data), self.n_dim)),
            tf.keras.layers.Conv1D(128, len(self.data), padding='same', activation="relu"),
            tf.keras.layers.Conv1D(512, len(self.data), padding='same', activation="relu"),
            tf.keras.layers.Dense(1),
        ]

        gen_opt = tf.keras.optimizers.Adam(0.001, beta_1=0.5, beta_2=0.9)
        disc_opt = tf.keras.optimizers.RMSprop(0.0005)

        class WGAN(tf.keras.Model):
            def __init__(self, gen, disc, gen_optimizer, disc_optimizer, n_Z, n_datasets, gp_weight=10.0):
                super().__init__()
                self.gen = tf.keras.Sequential(gen)
                self.disc = tf.keras.Sequential(disc)
                self.gen_optimizer = gen_optimizer
                self.disc_optimizer = disc_optimizer
                self.n_Z = n_Z
                self.n_datasets = n_datasets
                self.gp_weight = gp_weight

            def sample_z(self, n_samples):
                return tf.random.normal([n_samples, self.n_datasets, self.n_Z])

            def generate(self, n_samples=None, z=None):
                if z is None:
                    z = self.sample_z(n_samples)
                return self.gen(z)

            def compute_loss(self, x):
                x_gen = self.generate(n_samples=x.shape[0])

                logits_x = self.disc(x)
                logits_x_gen = self.disc(x_gen)

                epsilon = tf.random.uniform([x.shape[0], 1, 1])
                x_hat = epsilon * x + (1 - epsilon) * x_gen

                with tf.GradientTape() as t:
                    t.watch(x_hat)
                    d_hat = self.disc(x_hat)

                grad = t.gradient(d_hat, x_hat)
                gp = tf.reduce_mean((tf.sqrt(tf.reduce_sum(grad ** 2, axis=[1, 2])) - 1.0) ** 2)

                disc_loss = tf.reduce_mean(logits_x) - tf.reduce_mean(logits_x_gen) + gp * self.gp_weight
                gen_loss = tf.reduce_mean(logits_x_gen)

                return disc_loss, gen_loss

            @tf.function
            def train_step(self, x):
                with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                    disc_loss, gen_loss = self.compute_loss(x)

                g_grad = g_tape.gradient(gen_loss, self.gen.trainable_variables)
                d_grad = d_tape.gradient(disc_loss, self.disc.trainable_variables)

                self.gen_optimizer.apply_gradients(zip(g_grad, self.gen.trainable_variables))
                self.disc_optimizer.apply_gradients(zip(d_grad, self.disc.trainable_variables))

                return disc_loss, gen_loss

            def save_weights_custom(self, filepath):
                self.gen.save_weights(filepath + "_gen.weights.h5")
                self.disc.save_weights(filepath + "_disc.weights.h5")

        self.model = WGAN(generator, discriminator, gen_opt, disc_opt, self.N_Z, len(self.data))

    ################################
    # Training controller
    ################################
    def train(self):
        if self.mode == "global":
            self._train_one_round(batch_idx=None)
        else:
            n_batches = len(self.data[0].obs['batch_train'].cat.categories)
            for j in range(n_batches):
                print(f"\n⚡ Training batch {j}")
                self._train_one_round(batch_idx=j)

    ################################
    # Single training round
    ################################
    def _train_one_round(self, batch_idx=None):

        train_ds, _ = self.prepare_dataset(batch_idx)

        gen_loss_history, disc_loss_history = [], []

        for epoch in range(self.n_epochs):
            g_losses, d_losses = [], []

            for batch in train_ds:
                d_loss, g_loss = self.model.train_step(batch)
                g_losses.append(g_loss.numpy())
                d_losses.append(d_loss.numpy())

            mean_g = float(np.mean(g_losses))
            mean_d = float(np.mean(d_losses))

            gen_loss_history.append(mean_g)
            disc_loss_history.append(mean_d)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:05d} | D_loss: {mean_d:.4f} | G_loss: {mean_g:.4f}")

        # ---------- Save weights ----------
        suffix = "" if batch_idx is None else f"_batch_{batch_idx}"
        self.model.save_weights_custom(os.path.join(self.path, f"MOWGAN_model{suffix}"))

        # ---------- Save loss history ----------
        loss_path = os.path.join(self.path, f"loss_history{suffix}.pkl")
        with open(loss_path, "wb") as f:
            pickle.dump({"gen_loss": gen_loss_history, "disc_loss": disc_loss_history}, f)
        print(f"📉 Saved loss history -> {loss_path}")

        # ---------- Generate samples ----------
        self._generate_samples(batch_idx)

    ################################
    # Sample generation (training)
    ################################
    def _generate_samples(self, batch_idx=None):
        samples = self.model.generate(n_samples=self.n_samples).numpy()

        for i, ad in enumerate(self.data):
            key = f"scaler{i}" if batch_idx is None else f"scaler{i}_{batch_idx}"
            scaler = self.scalers[key]

            neigh = KNeighborsRegressor(n_neighbors=2)
            neigh.fit(ad.obsm[self.query[i]][:, :self.n_dim], ad.X)

            data_obsm = scaler.inverse_transform(samples[:, i, :])
            data_X = neigh.predict(data_obsm)

            ad_MOWGAN = anndata.AnnData(data_X)
            ad_MOWGAN.obsm[self.query[i]] = data_obsm
            ad_MOWGAN.var_names = ad.var_names

            if self.mode == "global":
                name = self.save_name[i] if self.save_name else f"data{i}_MOWGAN"
            else:
                name = f"anndata{i}_{batch_idx}"

            ad_MOWGAN.write(os.path.join(self.path, f"{name}.h5ad"))
            print(f"✅ Saved generated data: {name}.h5ad")

    ################################
    # Merge batch outputs
    ################################
    def merge_batches(self):
        if self.mode != "batch":
            print("Merge is only available in batch mode")
            return

        for i in range(len(self.data)):
            files = [f for f in os.listdir(self.path) if f.startswith(f"anndata{i}_")]
            ad_list = [sc.read(os.path.join(self.path, f)) for f in files]

            if not ad_list:
                continue

            merged = anndata.concat(ad_list, label='batch')
            save_name = self.save_name[i] if self.save_name else f"data{i}_MOWGAN"
            merged.write(os.path.join(self.path, f"{save_name}.h5ad"))
            print(f"✅ Merged {len(ad_list)} batches for dataset {i}")

    ################################
    # Save/load scalers
    ################################
    def save_scalers(self, batch_idx=None):
        suffix = "" if batch_idx is None else f"_batch_{batch_idx}"
        scaler_path = os.path.join(self.path, f"scalers{suffix}.pkl")
        os.makedirs(self.path, exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scalers, f)
        mode_str = "global" if batch_idx is None else f"batch {batch_idx}"
        print(f"✅ Saved scalers for {mode_str} -> {scaler_path}")

    def load_scalers(self, batch_idx=None):
        suffix = "" if batch_idx is None else f"_batch_{batch_idx}"
        scaler_path = os.path.join(self.path, f"scalers{suffix}.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                batch_scalers = pickle.load(f)
            self.scalers.update(batch_scalers)
            mode_str = "global" if batch_idx is None else f"batch {batch_idx}"
            print(f"✅ Loaded scalers for {mode_str} from {scaler_path}")
        else:
            print(f"⚠️ No scaler file found at {scaler_path}")

    ################################
    # Load saved weights
    ################################
    def load_weights_custom(self, batch_idx=None):
        suffix = "" if batch_idx is None else f"_batch_{batch_idx}"
        gen_path = os.path.join(self.path, f"MOWGAN_model{suffix}_gen.weights.h5")
        disc_path = os.path.join(self.path, f"MOWGAN_model{suffix}_disc.weights.h5")

        if not self.model.gen.built:
            dummy_z = tf.random.normal([1, self.model.n_datasets, self.model.n_Z])
            _ = self.model.gen(dummy_z)

        if not self.model.disc.built:
            dummy_x = tf.random.normal([1, self.model.n_datasets, self.model.gen.output_shape[-1]])
            _ = self.model.disc(dummy_x)

        self.model.gen.load_weights(gen_path)
        self.model.disc.load_weights(disc_path)
        mode_str = "global" if batch_idx is None else f"batch {batch_idx}"
        print(f"✅ Loaded generator & discriminator weights for {mode_str}")

    ################################
    # Fully reconstruct AnnData from loaded weights
    ################################
    def generate_and_construct_anndata(self, n_samples=None, batch_idx=None):
        n_samples = n_samples or self.n_samples

        # 1️⃣ Load weights and scalers
        self.load_weights_custom(batch_idx=batch_idx)
        self.load_scalers(batch_idx=batch_idx)

        # 2️⃣ Generate samples
        samples = self.model.generate(n_samples=n_samples).numpy()

        # 3️⃣ Reconstruct features and build AnnData
        for i, ad in enumerate(self.data):
            key = f"scaler{i}" if batch_idx is None else f"scaler{i}_{batch_idx}"
            scaler = self.scalers[key]

            # Inverse transform to embedding space
            data_obsm = scaler.inverse_transform(samples[:, i, :])

            # Map back to original feature space using KNN
            neigh = KNeighborsRegressor(n_neighbors=2)
            neigh.fit(ad.obsm[self.query[i]][:, :self.n_dim], ad.X)
            data_X = neigh.predict(data_obsm)

            # Construct AnnData
            ad_MOWGAN = anndata.AnnData(data_X)
            ad_MOWGAN.obsm[self.query[i]] = data_obsm
            ad_MOWGAN.var_names = ad.var_names

            # Determine save name
            if self.mode == "global":
                name = self.save_name[i] if self.save_name else f"data{i}_MOWGAN"
            else:
                name = f"anndata{i}_{batch_idx}"

            # Save
            ad_MOWGAN.write(os.path.join(self.path, f"{name}.h5ad"))
            print(f"✅ Saved generated data: {name}.h5ad")
