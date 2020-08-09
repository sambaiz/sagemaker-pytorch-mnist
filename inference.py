import torch


def predict_fn(input_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_data = input_data.to(device)
    model.eval()
    with torch.jit.optimized_execution(True, {"target_device": "eia:0"}):
        output = model(input_data)
        return output.max(1)[1]