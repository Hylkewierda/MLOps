import matplotlib.pyplot as plt
import json

with open("logs/losses.json") as f:
    data = json.load(f)

plt.plot(data["train"], label="Train Loss")
plt.plot(data["val"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss (PCAM)")
plt.legend()
plt.savefig("assets/training_loss.png")
plt.show()
