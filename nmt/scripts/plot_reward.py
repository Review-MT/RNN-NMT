import re
import matplotlib.pyplot as plt

# ============================================================
# Log files
# ============================================================
rt_file  = "rt"     # Random
bt_file  = "bt"     # BERT-Unfreeze
bfr_file = "bfr"    # BERT-Freeze

# ============================================================
# Extract Validation Reward
# ============================================================
def extract_validation_reward(log_file, max_epochs=10):

    rewards = []

    with open(log_file, "r") as f:
        lines = f.readlines()

    for line in lines:

        match = re.search(
            r"Validation sentence reward:\s*([0-9]*\.?[0-9]+)",
            line
        )

        if match:
            rewards.append(float(match.group(1)))

    return rewards[:max_epochs]

# ============================================================
# Load rewards
# ============================================================
rt_reward  = extract_validation_reward(rt_file)
bt_reward  = extract_validation_reward(bt_file)
bfr_reward = extract_validation_reward(bfr_file)

# ============================================================
# Epochs
# ============================================================
epochs_rt  = range(1, len(rt_reward) + 1)
epochs_bt  = range(1, len(bt_reward) + 1)
epochs_bfr = range(1, len(bfr_reward) + 1)

# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(12, 7))

# ------------------------------------------------------------
# Random
# ------------------------------------------------------------
plt.plot(
    epochs_rt,
    rt_reward,
    linestyle='-',
    marker='o',
    linewidth=3,
    markersize=8,
    label='Random'
)

# ------------------------------------------------------------
# BERT-Unfreeze
# ------------------------------------------------------------
plt.plot(
    epochs_bt,
    bt_reward,
    linestyle='--',
    marker='s',
    linewidth=3,
    markersize=8,
    label='BERT-Unfreeze'
)

# ------------------------------------------------------------
# BERT-Freeze
# ------------------------------------------------------------
plt.plot(
    epochs_bfr,
    bfr_reward,
    linestyle=':',
    marker='^',
    linewidth=4,
    markersize=8,
    label='BERT-Freeze'
)

# ============================================================
# Formatting
# ============================================================
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Validation Sentence Reward", fontsize=18)

plt.title(
    "Validation Reward Comparison(BLEU)",
    fontsize=22,
    fontweight='bold'
)

plt.xticks(range(1, 11), fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True, alpha=0.3)

plt.legend(
    fontsize=14,
    frameon=True
)

plt.tight_layout()

# ============================================================
# Save Figure
# ============================================================
plt.savefig(
    "validation_reward_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()
