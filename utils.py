import matplotlib.pyplot as plt
import seaborn as sns
import scipy
# =============================================================================
# Plots and Figures 
# =============================================================================
# Function to create Bland-Altman plots with R², ICC, and CoV annotation
def bland_altman_plot(ax, X, Y, title_str):
    """
    Create a Bland–Altman plot with equal axis scaling.
    """
    avg_val = (X + Y) / 2.0
    diff_val = Y - X

    sns.scatterplot(x=avg_val, y=diff_val, ax=ax, marker='o', edgecolor='none')
    ax.set_title(f'{title_str}', fontsize=12)

    mean_diff = np.mean(diff_val)
    std_diff = np.std(diff_val)
    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff

    ax.axhline(mean_diff, color='red', linestyle='--', linewidth=1.5)
    ax.axhline(loa_lower, color='black', linestyle='--')
    ax.axhline(loa_upper, color='black', linestyle='--')
    ax.set_xlabel('Average')
    ax.set_ylabel('Difference (Y - X)')

    # Ensure equal axis scaling
    x_min, x_max = avg_val.min(), avg_val.max()
    y_min, y_max = diff_val.min(), diff_val.max()

    axis_limit = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    ax.set_xlim([0, axis_limit])
    ax.set_ylim([-axis_limit, axis_limit])

    # Display R², ICC, and CoV instead of mean, LoA, and UpA
    slope, intercept, R_val, p_value, std_err = scipy.stats.linregress(X,Y)
    r_sq = R_val**2

    icc_data = np.concatenate((X,Y),axis=1)
    var = variation(icc_data, axis=1)
    cov_val = np.mean(var,axis=0)*100
    
    ax.text(axis_limit * 0.4, -axis_limit * 0.9,
            f"R² = {r_sq:.3f}\nCoV = {cov_val:.2f}%",
            fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
