import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np





class MotionData(Dataset):

    def __init__(self, filename):
        self.data = pickle.load(open(filename, "rb"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = torch.FloatTensor(item[0])
        state = torch.FloatTensor(item[1])
        action = torch.FloatTensor(item[2])
        return (image, state, action)


class CAE(nn.Module):

    def __init__(self):
        super(CAE, self).__init__()

        self.conv1 = nn.Conv2d(3, 5, 5, stride=1)
        self.conv2 = nn.Conv2d(5, 5, 5, stride=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(80, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 6)

        self.enc_fc1 = nn.Linear(20,30)
        self.enc_fc2 = nn.Linear(30,30)
        self.enc_fc3 = nn.Linear(30,1)

        self.dec_fc1 = nn.Linear(14,30)
        self.dec_fc2 = nn.Linear(30,30)
        self.dec_fc3 = nn.Linear(30,7)

        self.loss_func = nn.MSELoss()

    def image(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 80)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def encoder(self, x):
        img = self.image(x[0])
        x = torch.cat((img, x[1], x[2]), 1)
        h1 = torch.tanh(self.enc_fc1(x))
        h2 = torch.tanh(self.enc_fc2(h1))
        return self.enc_fc3(h2)

    def decoder(self, context):
        img = self.image(context[0])
        x = torch.cat((img, context[1], context[2]), 1)
        h1 = torch.tanh(self.dec_fc1(x))
        h2 = torch.tanh(self.dec_fc2(h1))
        return self.dec_fc3(h2)

    def forward(self, x):
        z = self.encoder(x)
        context = (x[0], x[1], z)
        a = self.decoder(context)
        loss = self.loss(a, x[2])
        return loss

    def loss(self, a_decoded, a_target):
        return self.loss_func(a_decoded, a_target)



def main():

    model = CAE()
    name = "CAE"

    EPOCH = 900
    BATCH_SIZE_TRAIN = 300
    LR = 0.01
    LR_STEP_SIZE = 300
    LR_GAMMA = 0.1

    dataname = "data/traj_dataset.pkl"
    savename = "models/" + name + "_model"

    train_data = MotionData(dataname)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        for batch, x in enumerate(train_set):
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
