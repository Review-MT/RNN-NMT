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
# Different line styles for each setup
# =====================================================
line_styles = {
    "Random": "-",
    "BERT-Unfreeze": "--",
    "BERT-Freeze": ":"
}

# =====================================================
# Colors
# Train and Validation use different colors
# =====================================================
train_colors = {
    "Random": "blue",
    "BERT-Unfreeze": "green",
    "BERT-Freeze": "red"
}

valid_colors = {
    "Random": "cyan",
    "BERT-Unfreeze": "lime",
    "BERT-Freeze": "orange"
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
        # -----------------------------------------
        train_match = re.search(
            r"Train Loss:\s*([0-9.]+)",
            line
        )

        if train_match:
            train_losses.append(float(train_match.group(1)))

        # -----------------------------------------
        # Validation Loss
        # -----------------------------------------
        valid_match = re.search(
            r"Valid Loss:\s*([0-9.]+)",
            line
        )

        if valid_match:
            valid_losses.append(float(valid_match.group(1)))

    return train_losses[:max_epochs], valid_losses[:max_epochs]


# =====================================================
# Plot
# =====================================================
plt.figure(figsize=(11,6))

for model_name, file_path in files.items():

    train_losses, valid_losses = extract_losses(file_path)

    epochs = range(1, len(train_losses) + 1)

    style = line_styles[model_name]

    # -------------------------------------------------
    # Training Loss
    # -------------------------------------------------
    plt.plot(
        epochs,
        train_losses,
        linestyle=style,
        marker='o',
        linewidth=2.5,
        color=train_colors[model_name],
        label=f"{model_name} Train"
    )

    # -------------------------------------------------
    # Validation Loss
    # -------------------------------------------------
    plt.plot(
        epochs,
        valid_losses,
        linestyle=style,
        marker='s',
        linewidth=2.5,
        color=valid_colors[model_name],
        label=f"{model_name} Valid"
    )

# =====================================================
# Labels and Formatting
# =====================================================
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Loss", fontsize=13)

plt.title(
    "Training and Validation Loss Comparison",
    fontsize=15
)

plt.xticks(range(1, 11))

plt.grid(True, alpha=0.3)

plt.legend(fontsize=10)

# =====================================================
# Save
# =====================================================
plt.savefig(
    "all_models_loss_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

# =====================================================
# Show
# =====================================================
plt.show()

print("Saved: all_models_loss_comparison.png")
