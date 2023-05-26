import matplotlib.pyplot as plt

def plot_solow_diagram(k: int, n: float, s: float, B: int, alpha: float, delta: float, kt_xmax: int, kt_vline: float):
    # Calculate the growth rates of capital per capita and the diagonal line
    k_growth = [s * B * t**alpha for t in range(kt_xmax + 1)]
    diagonal = [(n + delta) * t for t in range(kt_xmax + 1)]

    # Calculate steady state capital per capita
    k_star = ((s*B)/(n+delta))**(1/(1-alpha))

    # Configure plot settings
    plt.rcParams['axes.facecolor'] = 'white'
    plt.figure(figsize=(8, 6))
    plt.plot(k_growth, label=r'$sBk_t^{\alpha}$', color='darkorange')
    plt.plot(diagonal, label=r'$(n+\delta)k_t$', color='black')
    plt.axvline(x=k_star, linestyle='--', color='red', label=r'$k*$')
    plt.axvline(x=kt_vline, linestyle='--', color='green', label=r'$k_t$')
    plt.xlim(0, kt_xmax)
    plt.xlabel('Capital per capita, $k_t$')
    plt.ylabel('')
    plt.legend()
    plt.title('Figure 1: Solow Diagram')
    plt.grid(True)  # add grid
    
    # Add arrows to the plot
    arrow_len = abs(k_star - kt_vline)
    arrow_positions = [0.3, 0.6, 0.8, 0.9, 0.95, 1]
    
    # If kt_vline is to the left of k_star, flip the direction of the arrows
    if kt_vline < k_star:
        arrow_len = -arrow_len
        arrow_positions = [1 - pos for pos in arrow_positions]
    else:
        arrow_positions = [1 - pos for pos in arrow_positions]
    
    arrow_center = min(k_star, kt_vline) + abs(arrow_len) / 2
    
    # Add each arrow to the plot
    for pos in arrow_positions:
        arrow_x = arrow_center + arrow_len * (pos - 0.5)
        plt.annotate(
            '',
            xy=(kt_vline, 0), xycoords='data',
            xytext=(arrow_x, 0), textcoords='data',
            arrowprops=dict(
                arrowstyle="<-", color="black", lw=1, mutation_scale=15
            )
        )

    # Display the plot
    plt.show()
