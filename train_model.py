import os, csv, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class TurtleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(6656, 100), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.fc(self.flatten(self.features(x)))

class OvalDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        self.samples = []
        with open(os.path.join(data_dir, 'data.csv')) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.samples.append((row[0], float(row[2])))
        self.resize = transforms.Resize((120, 160))
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, ang_z = self.samples[idx]
        img = Image.open(os.path.join(self.data_dir, rel_path)).convert('RGB')
        img = self.resize(img)
        if self.is_train and (torch.rand(1).item() > 0.5):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            ang_z = -ang_z
        return self.norm(self.to_tensor(img)), torch.tensor([ang_z], dtype=torch.float32)

def train():
    data_dir = '/home/ubuntu/dataset'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Training on: {device}')
    ds = OvalDataset(data_dir)
    n_val = int(len(ds) * 0.2)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    model = TurtleNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print('[INFO] Starting 5 epochs...')
    for epoch in range(5):
        model.train()
        t_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                v_loss += criterion(model(x), y).item()
        print(f'Epoch {epoch+1}/5 | Train Loss: {t_loss/len(train_loader):.5f} | Val Loss: {v_loss/len(val_loader):.5f}')

    out_path = '/home/ubuntu/auto_drive_model.pth'
    torch.save(model.state_dict(), out_path)
    print(f'[SUCCESS] Model saved to {out_path}')

if __name__ == '__main__':
    train()