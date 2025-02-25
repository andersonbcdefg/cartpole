from datetime import datetime
import matplotlib.pyplot as plt

returns = []  # Track returns across episodes

from datetime import datetime
import matplotlib.pyplot as plt


def save_graph(returns, value_losses, means=None, stds=None):
    """
    Save training graphs showing returns and losses over time.
    Optionally includes policy distribution statistics if provided.

    Args:
        returns (list): Episode returns over time
        value_losses (list): Value network losses over time
        means (list, optional): Mean action values over time
        stds (list, optional): Standard deviations over time
    """
    if means is not None or stds is not None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Returns plot
    ax1.plot(returns)
    ax1.set_title('Average Returns Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Return')

    # Value loss plot
    ax2.plot(value_losses)
    ax2.set_title('Value Network MSE Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('MSE Loss')
    ax2.set_yscale('log')

    if means is not None:
        ax3.plot(means)
        ax3.set_title('Policy Mean Over Time')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Action Mean')

    if stds is not None:
        ax4.plot(stds)
        ax4.set_title('Policy Standard Deviation Over Time')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Action Std Dev')

    timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M')
    filename = f'training_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved training graphs to {filename}")
