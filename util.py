from datetime import datetime
import matplotlib.pyplot as plt

returns = []  # Track returns across episodes

def save_graph(returns, value_losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(returns)
    ax1.set_title('Average Returns Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Return')

    ax2.plot(value_losses)
    ax2.set_title('Value Network MSE Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('MSE Loss')
    ax2.set_yscale('log')  # Often helpful for loss plots

    timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M')
    filename = f'training_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved training graphs to {filename}")
