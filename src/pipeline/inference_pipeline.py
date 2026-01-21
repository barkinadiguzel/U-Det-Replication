import torch
from udet import UDet
from luna_loader import LUNADataset
from torch.utils.data import DataLoader

def inference(model_path, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UDet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    results = []
    with torch.no_grad():
        for img, _ in loader:
            img = img.to(device)
            pred = model(img)
            results.append(pred.cpu().numpy())
    return results
