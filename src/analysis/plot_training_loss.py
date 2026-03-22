import re
import matplotlib.pyplot as plt

log_file = "finetuning/training_log.txt"

losses = []
steps = []

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(r"Loss:\s*([0-9.]+)", line)
        if match:
            loss = float(match.group(1))
            losses.append(loss)
            steps.append(len(losses))

if not losses:
    print("No loss values found in training_log.txt")
    exit()

plt.figure(figsize=(8, 5))
plt.plot(steps, losses, marker="o")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("finetuning/training_loss_plot.png", dpi=300)
plt.show()

print("Saved as finetuning/training_loss_plot.png")