import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import numpy as np
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

    
    icc_data = np.concatenate((X.reshape(-1,1),Y.reshape(-1,1)),axis=1)
    var = scipy.stats.variation(icc_data, axis=1)
    cov_val = np.mean(var,axis=0)*100
    icc_val = icc(icc_data,icc_type='ICC(3,1)')
    ax.text(axis_limit * 0.4, -axis_limit * 0.9,
            f"R² = {r_sq:.3f}\nICC = {icc_val:.3f}\nCoV = {cov_val:.2f}%",
            fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

def icc(Y, icc_type='ICC(2,1)'):
    ''' Calculate intraclass correlation coefficient

    ICC Formulas are based on:
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
    assessing rater reliability. Psychological bulletin, 86(2), 420.
    icc1:  x_ij = mu + beta_j + w_ij
    icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij
    Code modifed from nipype algorithms.icc
    https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py

    Args:
        Y: The data Y are entered as a 'table' ie. subjects are in rows and repeated
            measures in columns
        icc_type: type of ICC to calculate. (ICC(2,1), ICC(2,k), ICC(3,1), ICC(3,k)) 
    Returns:
        ICC: (np.array) intraclass correlation coefficient
    '''

    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k-1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))),
                                X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc  # / n (without n in SPSS results)

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == 'icc1':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)
        NotImplementedError("This method isn't implemented yet.")

    elif icc_type == 'ICC(2,1)' or icc_type == 'ICC(2,k)':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        if icc_type == 'ICC(2,k)':
            k = 1
        ICC = (MSR - MSE) / (MSR + (k-1) * MSE + k * (MSC - MSE) / n)

    elif icc_type == 'ICC(3,1)' or icc_type == 'ICC(3,k)':
        # ICC(3,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error)
        if icc_type == 'ICC(3,k)':
            k = 1
        ICC = (MSR - MSE) / (MSR + (k-1) * MSE)

    return ICC