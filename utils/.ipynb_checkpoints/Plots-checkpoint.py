# These codes has been adapted from Claudio, Fanconi & Anne, de Hond & Dylan, Peterson & Angelo, Capodici & Tina, Hernandez-Boussard (2023)
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def predictions_with_uncertainty(
    predictive_distribution,
    true_labels,
    title,
    method,
    t: float = 0.5,
    std_factor: float = 1.0,
    use_quantile: bool = True,
    quantile: float = 0.95):
    
    # Convert predictive distribution to DataFrame statistics
    predicted_values = predictive_distribution.describe().T
    predicted_values = predicted_values.drop("count", axis=1)
    predicted_values["true_label"] = np.array(true_labels)
    predicted_values["id"] = np.arange(predicted_values.shape[0])

    # Calculate the lower and upper bounds
    predicted_values['low'] = predicted_values['mean'] - predicted_values['std']
    predicted_values['upper'] = predicted_values['mean'] + predicted_values['std']
    
    # Clip values to ensure they remain within [0,1]
    predicted_values['low'] = np.clip(predicted_values['low'], 0, 1)
    predicted_values['upper'] = np.clip(predicted_values['upper'], 0, 1)

    # Determine predicted labels based on threshold t
    predicted_values["predicted_label"] = (predicted_values["mean"] >= t).astype(int)
    
    # Determine correctness: Green for correct, Red for incorrect
    predicted_values["correct"] = predicted_values["predicted_label"] == predicted_values["true_label"]
    colors = {True: 'green', False: 'red'}

    # Compute classification accuracy
    accuracy = predicted_values["correct"].mean()

    # Sort values based on mean prediction
    sorted_predicted_values = predicted_values.sort_values("mean").reset_index()

    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3,5), sharex=True, sharey=True, layout="constrained")
    
    # Plot uncertainty ranges (error bars) on the first y-axis
    for index, row in sorted_predicted_values.iterrows():
        ax1.plot([index, index], [row['low'], row['upper']],
                color=colors[row["correct"]], alpha=0.3, lw = 1)
    
    # Plot the points on the first y-axis
    ax1.scatter(sorted_predicted_values.index, sorted_predicted_values['mean'],
               c=sorted_predicted_values['correct'].map(colors), s=1)
    
    # Set axis limits
    #ax1.set_xlim(0, len(sorted_predicted_values))
    
    # Add a threshold line
    ax1.axhline(
    y=t,  # The y-coordinate (height) of the horizontal line
    color='black',  # Color of the line
    linestyle='-',  # Line style (solid, dashed, etc.)
    linewidth=1,
    alpha = 0.5
    )
    #ax1.hlines(t, xmin=0, xmax=len(sorted_predicted_values), colors="black", linestyles="-", label = f"Accuracy: {accuracy:.2f}%")
    
    # Create custom legend handles for misclassification
    custom_lines = [Line2D([0], [0], color='black', lw=1, alpha = 0.5, label=f"Accuracy: {accuracy:.2f}"),
                    Line2D([0], [0], color='green', lw=1, alpha = 0.5, label='Correctly Classified'),
                    Line2D([0], [0], color='red', lw=1, alpha = 0.5, label='Misclassified')]

    #ax1.legend(loc='upper left', fontsize=8)
    
    ax1.legend(handles=custom_lines, loc='upper left', fontsize=6)
    #ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)
   
    
    # Calculate mean and uncertainty for the second y-axis
    mean = predictive_distribution.mean(0)
    sort = np.argsort(mean)
    mean = mean[sort]

    if use_quantile:
        quantified_uncertainty_string = f"{int(quantile*100)}%-CI"
        low_quantile = (1 - quantile) / 2  # Corrected
        high_quantile = 1 - low_quantile    # Corrected
        high = np.quantile(
            predictive_distribution, low_quantile, axis=0, method="inverted_cdf"
        )
        low = np.quantile(
            predictive_distribution, high_quantile, axis=0, method="inverted_cdf"
        )
        high = high[sort]
        low = low[sort]
        std = np.vstack([np.clip(mean - low, 0, None), np.clip(high - mean, 0, None)])

    else:
        quantified_uncertainty_string = (
            str(int(std_factor)) + "$\sigma$" if std_factor != 1 else "$\sigma$"
        )
        std = std_factor * predictive_distribution.std(0)
        std = std[sort]
        high = np.clip(mean + std, None, 1) 
        low = np.clip(mean - std, 0, None)
        std = np.vstack([np.clip(mean - low, 0, None), np.clip(high - mean, 0, None)])

    mask = ~((high > t) & (low < t))

    ax2.axhline(
    y=t,  # The y-coordinate (height) of the horizontal line
    color='black',  # Color of the line
    linestyle='-',  # Line style (solid, dashed, etc.)
    linewidth=1,
    alpha = 0.5, # Thickness of the line
    label=f"Certainty={mask.mean():.2f}"
    )
    
    # Plot uncertainty on the second y-axis
    ax2.errorbar(
        np.arange(len(mean))[~mask],
        mean[~mask],
        yerr=std[:, ~mask],
        lw = 1,
        marker=".",
        markersize = 1,
        color = "#1a80bb" if method == "FLR" else "#FF8C00",
        alpha=0.6,
        label=f"$\mu$ $\pm $ {quantified_uncertainty_string} High uncertainty",
        ls="none",
    )
    ax2.errorbar(
        np.arange(len(mean))[mask],
        mean[mask],
        yerr=std[:, mask], 
        lw = 1,
        marker=".",
        markersize = 1,
        color = "#1a80bb" if method == "FLR" else "#FF8C00",
        alpha=0.15,
        label=f"$\mu$ $\pm $ {quantified_uncertainty_string} Low uncertainty",
        ls="none",
    )

    # Labels for the second y-axis
    ax2.tick_params(axis='y', labelsize=8)
    
    # Add legend for uncertainty
    ax2.legend(loc='upper left', fontsize=6)
    
    # Remove spines for the second y-axis
    ax2.spines["right"].set_color("none")
    ax2.spines["top"].set_color("none")
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    fig.supylabel("Predicted probability (mean $\pm$ sd) ", fontsize=8)
    fig.suptitle(title, fontsize = 8)
    plt.xticks(fontsize = 8)
    plt.xlabel("Observations sorted by mean \n predicted probability", fontsize=8)
    for pos in ['right', 'top']: 
        plt.gca().spines[pos].set_visible(False) 
    
    return plt

# """
# Logistic regression curve - Classification
# Mean values of every predictive distribution is calculated and sorted.
# Standard deviation is calculated of every prediction.
# Mean +/- standard deviation the error bar (Uncertainty) is calculated. 
# """
# def sorted_predictions_classification(
#     predictive_distribution,
#     true_labels,
#     title,
#     t: float = 0.5):
#     predicted_values = predictive_distribution.describe().T
#     predicted_values = predicted_values.drop("count", axis=1)
#     predicted_values["true_label"] = np.array(true_labels)
#     predicted_values["id"] = np.arange(predicted_values.shape[0])

#     # Calculate the lower and upper bounds
#     predicted_values['low'] = predicted_values['mean'] - predicted_values['std']
#     predicted_values['upper'] = predicted_values['mean'] + predicted_values['std']
#     # Apply the bounds
#     predicted_values['low'] = predicted_values['low'].apply(lambda x: max(0, x))
#     predicted_values['upper'] = predicted_values['upper'].apply(lambda x: min(1, x))

#     sorted_predicted_values = predicted_values.sort_values("mean").reset_index()

#     # Set up the figure and axis
#     fig, ax = plt.subplots()
    
#     fig.set_figwidth(5)
#     fig.set_figheight(5)
    
#     # Create a color dictionary for true_label values
#     colors = {0: 'tab:blue', 1: 'tab:red'}
    
#     # # Plot the segments
#     for index, row in sorted_predicted_values.iterrows():
#         ax.plot([index, index], [row['low'], row['upper']],
#                 color=colors[row['true_label']], alpha=0.3, label=row['true_label'])
    
#     # Create custom legend handles
#     custom_lines = [Line2D([0], [0], color=colors[0], lw=2, label='0 Non-PE'),
#                     Line2D([0], [0], color=colors[1], lw=2, label='1 - PE')]
    
#     # Plot the points
#     ax.scatter(sorted_predicted_values.index, sorted_predicted_values['mean'],
#                c=sorted_predicted_values['true_label'].map(colors), s = 1)
    
#     # Set axis limits
#     ax.set_xlim(0, len(sorted_predicted_values))
    
#     # Add the custom legend
#     ax.legend(handles=custom_lines,loc='upper left')
#     plt.xlabel("Index", fontdict = {"fontsize": 8})
#     plt.ylabel("Prediction", fontdict = {"fontsize": 8})
#     plt.xticks(fontsize = 8)
#     plt.yticks(fontsize = 8)
#     plt.hlines(t,xmin=0, xmax=len(sorted_predicted_values), colors="black")
#     plt.title(title, fontdict = {"fontsize": 8})
#     return plt

# """
# Logistic regression curve - Uncertainty quantification
# Mean values of every predictive distribution is calculated and sorted.
# Standard deviation is calculated of every prediction.
# Mean +/- standard deviation the error bar (Uncertainty) is calculated. 
# High uncertainty and low uncertainty is calculated, if the error bar crosses the threshold.
# """

# def sorted_predictions_with_threshold(
#     predictive_distribution,
#     title,
#     t: float = 0.5,
#     std_factor: float = 1.0,
#     use_quantile: bool = True,
#     quantile: float = 0.95):
    
#     mean = predictive_distribution.mean(0)
#     sort = np.argsort(mean)
#     mean = mean[sort]

#     if use_quantile:
#         quantified_uncertainty_string = f"{int(quantile*100)}%-CI"
#         low_quantile = (1 - quantile) / 2  # Corrected
#         high_quantile = 1 - low_quantile    # Corrected
#         high = np.quantile(
#             predictive_distribution, low_quantile, axis=0, method="inverted_cdf"
#         )
#         low = np.quantile(
#             predictive_distribution, high_quantile, axis=0, method="inverted_cdf"
#         )
#         high = high[sort]
#         low = low[sort]
#         std = np.vstack([np.clip(mean - low, 0, None), np.clip(high - mean, 0, None)])

#     else:
#         quantified_uncertainty_string = (
#             str(int(std_factor)) + "$\sigma$" if std_factor != 1 else "$\sigma$"
#         )
#         std = std_factor * predictive_distribution.std(0)
#         std = std[sort]
#         high = np.clip(mean + std, None, 1) 
#         low = np.clip(mean - std, 0, None)
#         std = np.vstack([np.clip(mean - low, 0, None), np.clip(high - mean, 0, None)])

#     mask = ~((high > t) & (low < t))
#     plt.figure(figsize = (5,5))
#     plt.axhline(
#     y=t,  # The y-coordinate (height) of the horizontal line
#     color='black',  # Color of the line
#     linestyle='-',  # Line style (solid, dashed, etc.)
#     linewidth=1,  # Thickness of the line
#     label=f"Threshold (t={t}, certainty={mask.mean():.2f})"
#     )
#     mean_string = "$\\bar{y}$"
#     plt.errorbar(
#         np.arange(len(mean))[~mask],
#         mean[~mask],
#         yerr=std[:, ~mask],
#         marker=".",
#         color = "#d73027",
#         alpha=0.3,
#         label=f"uncertainty ({quantified_uncertainty_string}): High uncertainty",
#         ls="none",
#     )
#     plt.errorbar(
#         np.arange(len(mean))[mask],
#         mean[mask],
#         yerr=std[:, mask], 
#         marker=".",
#         color = "#a6d96a",
#         alpha=0.5,
#         label=f"uncertainty ({quantified_uncertainty_string}): Less uncertainty",
#         ls="none",
#     )

#     plt.ylabel("Predicted probability (mean $\pm$ sd) ", fontdict = {"fontsize": 8})
#     plt.xlabel("Observations sorted on mean \n predicted probability", fontdict = {"fontsize": 8})
#     plt.xticks(fontsize = 8)
#     plt.yticks(fontsize = 8)
#     plt.legend(fontsize = 8)
#     plt.gca().spines["right"].set_color("none")
#     plt.gca().spines["top"].set_color("none")
#     plt.title(title, fontdict = {"fontsize": 8})
#     # plt.grid()
#     return plt