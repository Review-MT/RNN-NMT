import re
import matplotlib.pyplot as plt

# =====================================================
# Log files
# =====================================================
files = {
    "Random": "rt",
    "BERT-Unfreeze": "bt",
    "BERT-Freeze": "bfr"
}

# =====================================================
# Function to extract losses
# =====================================================
def extract_losses(log_file, max_epochs=10):

    train_losses = []
    valid_losses = []

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:

        # -----------------------------------------
        # Train Loss
        # Example:
        # Train Loss: 2.0094 | Train PPL: 7.46
        # -----------------------------------------
        train_match = re.search(
            r"Train Loss:\s*([0-9.]+)",
            line
        )

        if train_match:
            train_losses.append(float(train_match.group(1)))

        # -----------------------------------------
        # Validation Loss
        # Example:
        # Valid Loss: 0.9234 | Valid PPL: 2.52
        # -----------------------------------------
        valid_match = re.search(
            r"Valid Loss:\s*([0-9.]+)",
            line
        )

        if valid_match:
            valid_losses.append(float(valid_match.group(1)))

    # Keep only first 10 epochs
    train_losses = train_losses[:max_epochs]
    valid_losses = valid_losses[:max_epochs]

    return train_losses, valid_losses


# =====================================================
# Plot
# =====================================================
plt.figure(figsize=(10,6))

for model_name, file_path in files.items():

    train_losses, valid_losses = extract_losses(file_path)

    epochs = range(1, len(train_losses) + 1)

    # -----------------------------------------
    # Training Loss
    # -----------------------------------------
    plt.plot(
        epochs,
        train_losses,
        marker='o',
        linewidth=2,
        label=f"{model_name} Train"
    )

    # -----------------------------------------
    # Validation Loss
    # -----------------------------------------
    plt.plot(
        epochs,
        valid_losses,
        marker='s',
        linewidth=2,
        linestyle='--',
        label=f"{model_name} Valid"
    )

# =====================================================
# Labels and formatting
# =====================================================
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Comparison")

plt.xticks(range(1, 11))

plt.grid(True)
plt.legend()

# =====================================================
# Save
# =====================================================
plt.savefig("all_models_loss_comparison.png", dpi=300)

# =====================================================
# Show
# =====================================================
plt.show()

print("Saved: all_models_loss_comparison.png")
