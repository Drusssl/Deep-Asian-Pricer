import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from generator import AsianBasketSimulator
from model import PricingNet, init_weights

def main():
    # 1. Data Generation
    print("Generating synthetic market data using Monte Carlo...")
    sim = AsianBasketSimulator(n_assets=4)
    X_np, Y_np = sim.generate_data(n_samples=50000, n_steps=50)
    
    # Convert to Tensors
    X = torch.from_numpy(X_np)
    Y = torch.from_numpy(Y_np)
    
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PricingNet(input_dim=5).to(device)
    model.apply(init_weights)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # 3. Training Loop
    epochs = 10
    print(f"Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | MSE Loss: {avg_loss:.6f}")
        
    # 4. Save Model
    torch.save(model.state_dict(), "asian_option_model.pth")
    print("Model saved to asian_option_model.pth")

if __name__ == "__main__":
    main()