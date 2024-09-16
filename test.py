import torch
import pandas as pd
from torch.utils.data import Dataset
import os
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from convlstm import ConvLSTM

os.environ['CUDA_VISIBLE_DEVICE'] = '0'
torch.cuda.set_device('cuda:0')

# load model
# model = torch.load("D:\code_ai_ecco\model.pth", map_location="cuda:0")
model = torch.load("D:\code_ai_ecco\model_test.pth", map_location="cuda:0")

class sshdataset(Dataset):
    def __init__(self, data, timestep):
        """
        Args:
            data (tensor): (idx,h,w)
            timestep (int): timesequence
        """
        self.data = data
        self.timestep = timestep

    def __getitem__(self, index):
        x = self.data[index:index+self.timestep, :, :]
        x = x.unsqueeze(1).cuda()
        y = self.data[index+self.timestep, :, :].unsqueeze(0).cuda()
        return x, y

    def __len__(self):
        length = self.data.shape[0] - self.timestep
        return max(0, length)

criterion = torch.nn.MSELoss()
model.eval()

for year in range(2004, 2017):
    # create dataset instance
    start = datetime(year-1, 12, 22).date()
    end = datetime(year, 12, 31).date()
    feature = []
    current_date = start

    while current_date<=end:
        file = f"F:\code_ai_ecco\ecco_v4r4\ssh_excel\SEA_SURFACE_HEIGHT_day_mean_{current_date}_ECCO_V4r4_latlon_0p50deg.nc.csv"
        ssh = pd.read_csv(file, header=None).values
        ssh[pd.isnull(ssh)] = 0
        feature.append(torch.Tensor(ssh))
        current_date = current_date+timedelta(days=1)  # 更新日期，每次增加一天

    if len(feature) == 0:
        continue

    alldate = torch.stack(feature, dim=0)
    dataset = sshdataset(alldate, 10)
    loader = DataLoader(dataset=dataset, batch_size=1)  # 将batch_size设置为1

    # Create a directory to save results
    result_dir = f"results/{year}"
    os.makedirs(result_dir, exist_ok=True)

    # Test and save results for each day
    with torch.no_grad():
        loss_list = []  # 用于保存每个日期的loss值
        for idx, (x, y) in enumerate(loader):
            out = model(x)
            loss = criterion(y, out)
            loss_list.append(loss.item())  # 将loss值添加到loss_list中

            # Save y (ground truth) for each day
            result_date = start + timedelta(days=idx+10)
            result_file = f"{result_dir}/{result_date}.csv"
            out_output = out.squeeze(0).cpu().numpy()
            out_output_2d = out_output.reshape(-1, out_output.shape[-1])
            pd.DataFrame(out_output_2d).to_csv(result_file, index=False)

            print(f"{result_date}: loss is {loss.item()}")

        # Save loss_list to CSV file
        loss_file = f"{result_dir}/{year}_loss.csv"
        pd.DataFrame(loss_list).to_csv(loss_file, index=False)

# Rest of the code...