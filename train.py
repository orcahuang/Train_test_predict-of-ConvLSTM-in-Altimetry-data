import torch
import pandas as pd
from torch.utils.data import Dataset
import os
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from convlstm import ConvLSTM

os.environ['CUDA_VISIBLE_DEVICE'] = '0'
torch.cuda.set_device('cuda:0')

# 定义数据集
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
        return self.data.shape[0] - self.timestep - 1

# 把数据导入内存
start = datetime(1993, 1, 1).date()
end = datetime(2014, 5, 1).date()
feature = []
while start != end:
    file = f"D:\data_for_ai\SLA\ssh_aviso_daily_excel_nan_replace\dt_global_allsat_phy_l4_{start}.csv"
    ssh = pd.read_csv(file, header=None).values
    ssh[pd.isnull(ssh)] = 0
    feature.append(torch.Tensor(ssh))
    start = start + timedelta(days=1)
alldate = torch.stack(feature, dim=0)

# 创建数据集和模型实例
model = ConvLSTM(input_dim=1, hidden_dim=[2, 1], kernel_size=(3, 3), output_dim=1, num_layers=2, batch_first=True).cuda()
dataset = sshdataset(alldate, 10)

# 加载已经训练好的模型
pretrained_model_path = "D:\code_ai_ecco\model_ecco_ssh_daily_nanreplace_19932014_lr1e3_epoch50_origin_data-20.pth"
pretrained_model = torch.load(pretrained_model_path)
model.load_state_dict(pretrained_model.state_dict())

# 继续训练
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 580
for epoch in range(epochs):
    print(f'第{epoch}个epoch训练')
    batch = 1
    for x, y in DataLoader(dataset=dataset, batch_size=8, shuffle=True):
        output = model(x)
        optimizer.zero_grad()
        loss = criterion(y, output)
        loss.backward()
        optimizer.step()
        print(f'第{batch}次训练,loss为{loss}')
        batch += 1
        import torch, gc

        gc.collect()
        torch.cuda.empty_cache()
        model_name = f"D:\code_ai_ecco\model_ecco_ssh_daily_19932014_lr1e3_epoch600_origin_data-{epoch}.pth"
        torch.save(model, model_name)  # 保存单个模型
# 保存整个模型
model_name = "D:\code_ai_ecco\model_ecco_ssh_daily_19932014_lr1e3_epoch600_origin_data.pth"
torch.save(model, model_name)