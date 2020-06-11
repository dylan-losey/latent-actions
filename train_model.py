import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import sys


# collect dataset
class MotionData(Dataset):

  def __init__(self, filename):
    self.data = pickle.load(open(filename, "rb"))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return torch.FloatTensor(self.data[idx])


# conditional autoencoder
class CAE(nn.Module):

  def __init__(self):
    super(CAE, self).__init__()

    self.name = "CAE"
    # state-action pair is size 14
    # latent space is size 2
    self.fc1 = nn.Linear(14,30)
    self.fc2 = nn.Linear(30,30)
    self.fc3 = nn.Linear(30,2)

    # state is size 7, latent space is size 2
    self.fc4 = nn.Linear(9,30)
    self.fc5 = nn.Linear(30,30)
    self.fc6 = nn.Linear(30,7)

    self.loss_func = nn.MSELoss()

  def encoder(self, x):
    h1 = torch.tanh(self.fc1(x))
    h2 = torch.tanh(self.fc2(h1))
    return self.fc3(h2)

  def decoder(self, z_with_state):
    h4 = torch.tanh(self.fc4(z_with_state))
    h5 = torch.tanh(self.fc5(h4))
    return self.fc6(h5)

  def forward(self, x):
    s = x[:, 0:7]
    a_target = x[:, 7:14]
    z = self.encoder(x)
    z_with_state = torch.cat((z, s), 1)
    a_decoded = self.decoder(z_with_state)
    loss = self.loss(a_decoded, a_target)
    return loss

  def loss(self, a_decoded, a_target):
    return self.loss_func(a_decoded, a_target)


# train cAE
def main():

    model = CAE()
    dataname = "data/traj_dataset.pkl"
    savename = "models/" + model.name + "_model"

    EPOCH = 600
    BATCH_SIZE_TRAIN = 400
    LR = 0.01
    LR_STEP_SIZE = 280
    LR_GAMMA = 0.1

    train_data = MotionData(dataname)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):
            optimizer.zero_grad()
            a_target = x[:, 7:14]
            loss = model(x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
