import numpy as np
import torch as T

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class VAE(T.nn.Module):
    def __init__(self, data_dim):
        super(VAE, self).__init__()
        self.fc1 = T.nn.Linear(data_dim, 16)
        self.fc2a = T.nn.Linear(16, 2)  # u
        self.fc2b = T.nn.Linear(16, 2)  # log-var
        self.fc3 = T.nn.Linear(2, 16)
        self.fc4 = T.nn.Linear(16, data_dim)

    def encode(self, x):
        z = T.relu(self.fc1(x))
        z1 = self.fc2a(z)
        z2 = self.fc2b(z)
        return (z1, z2)  # (u, log-var)

    def decode(self, x):
        z = T.relu(self.fc3(x))
        z = T.sigmoid(self.fc4(z))
        return z

    def sampling(self, u, logvar):
        stdev = T.exp(0.5 * logvar)
        noise = T.randn_like(stdev)
        sample = u + (noise * stdev)
        return sample

    def forward(self, x):
        (u, logvar) = self.encode(x)
        z = self.sampling(u, logvar)
        oupt = self.decode(z)
        return (oupt, u, logvar)

def cus_loss_func(recon_x, x, u, logvar):
    mse = T.nn.functional.mse_loss(recon_x, x)
    kld = -0.5 * T.sum(1 + logvar - u.pow(2) - logvar.exp())
    BETA = 5
    return mse + (BETA * kld), mse


def train(vae, data, batch_size, max_epochs, lr):
    data_ldr = T.utils.data.DataLoader(data, batch_size=batch_size,
                                       shuffle=True)

    opt = T.optim.SGD(vae.parameters(), lr=lr)
    print("Starting training")
    for epoch in range(max_epochs):
        for (b_idx, batch) in enumerate(data_ldr):
            opt.zero_grad()
            X = batch
            recon_x, u, logvar = vae(X)
            loss_val,mse = cus_loss_func(recon_x, X, u, logvar)
            loss_val.backward()
            opt.step()

        if epoch != 0 and epoch % 2 == 0:
            print("epoch = %6d" % epoch, end="")
            print("  curr batch loss = %7.4f" % loss_val.item(), end="")
            print("  curr mse loss = %7.4f" % mse.item(), end="")
            print("")

    print("Training complete ")


def main():
    data = np.load("./results/TD3_Walker2d-v2_0_states.npy")
    data /= 10.0
    data = T.tensor(data, dtype=T.float32).to(device)

    vae = VAE(data.shape[1]).to(device)
    vae.train()

    batch_size = 256
    max_epochs = 32
    lr = 0.001
    train(vae, data, batch_size, max_epochs, lr)

    T.save(vae.state_dict(), "./models/vae")

    vae.eval()
    with T.no_grad():
        u, logvar = vae.encode(data)

    u = u.cpu().data.numpy()
    logvar = logvar.cpu().data.numpy()
    res = [u, logvar]
    np.save(f"./results/vae_2d", res)


if __name__ == "__main__":
    main()
