
import torch

def train(model, dataloader, epochs=5):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x, features, y in dataloader:
            optimizer.zero_grad()

            preds = model(x, features)
            loss = criterion(preds.squeeze(), y.float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
