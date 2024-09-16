import torch
import pandas as pd
from torch.utils.data import Dataset
import os
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from convlstm import ConvLSTM
import numpy as np

os.environ['CUDA_VISIBLE_DEVICE'] = '0'
torch.cuda.set_device('cuda:0')

# Load the model
model = torch.load("D:\code_ai_ecco\model_ecco_ssh_daily_19932014_lr1e3_epoch600_origin_data-570.pth", map_location="cuda:0")

class sshdataset(Dataset):
    def __init__(self, data, timestep):
        """
        Args:
            data (tensor): (idx, h, w)
            timestep (int): timesequence
        """
        self.data = data
        self.timestep = timestep

    def __getitem__(self, index):
        x = self.data[index:index+self.timestep, :, :]
        x = x.unsqueeze(1).cuda()
        return x

    def __len__(self):
        length = self.data.shape[0] - self.timestep
        return max(0, length)

def predict_and_save_results(model, dataset, loader, result_dir, start_date):
    with torch.no_grad():
        predictions = []  # Store the predictions
        prev_predictions = []  # Store the most recent predictions
        for idx, x in enumerate(loader):
            if x is not None:
                out = model(x)
                predictions.append(out.squeeze(0).cpu().numpy())
                prev_predictions.append(out.squeeze(0).cpu().numpy())
            else:
                if len(prev_predictions) > 10:  # Use the average of the 10 most recent predictions
                    prev_prediction = np.mean(prev_predictions[-10:], axis=0)
                    predictions.append(prev_prediction)
                    prev_predictions.append(prev_prediction)
                else:
                    predictions.append(None)
                    prev_predictions.append(None)

        # Save the predictions to CSV files
        current_date = start_date
        for prediction in predictions:
            if prediction is not None:
                predict_date = current_date - timedelta(days=365)
                result_file = f"{result_dir}/d_{predict_date}.csv"
                prediction_2d = np.squeeze(prediction).reshape(-1, prediction.shape[-1])
                df = pd.DataFrame(prediction_2d)
                df.to_csv(result_file, index=False)
            else:
                # Use the average of the 10 most recent predictions
                prev_prediction = np.mean(prev_predictions[-10:], axis=0)
                predict_date = current_date - timedelta(days=365)
                result_file = f"{result_dir}/d_{predict_date}.csv"
                prediction_2d = np.squeeze(prev_prediction).reshape(-1, prev_prediction.shape[-1])
                df = pd.DataFrame(prediction_2d)
                df.to_csv(result_file, index=False)
            current_date += timedelta(days=1)

        print("Prediction completed!")

# Create dataset instance
start_date = datetime(2017, 1, 1).date()
end_date = datetime(2017, 12, 31).date()
predict_end_date = end_date + timedelta(days=730)  # Predict for 60 days
feature = []
current_date = start_date - timedelta(days=1)  # Start from the day before the start date

# Load data for prediction
while current_date <= predict_end_date:
    file = f"D:\data_for_ai\predict_start\year5\d_{current_date}.csv"
    if os.path.exists(file):
        ssh = pd.read_csv(file, header=None).values
        ssh[pd.isnull(ssh)] = 0
        feature.append(torch.Tensor(ssh))
    else:
        feature.append(None)  # Append None if the file doesn't exist
    current_date += timedelta(days=1)

# Filter out None values
filtered_feature = [x for x in feature if x is not None]

if len(filtered_feature) > 0:
    alldate = torch.stack(filtered_feature, dim=0)
    dataset = sshdataset(alldate, 1)
    loader = DataLoader(dataset=dataset, batch_size=1)

    # Create a directory to save results
    result_dir = f"results/predictions"
    os.makedirs(result_dir, exist_ok=True)

    # Predict and save the results
    predict_and_save_results(model, dataset, loader, result_dir, end_date + timedelta(days=1))
else:
    print("No valid data found for prediction.")