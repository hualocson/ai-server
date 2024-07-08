# Import necessary libraries
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# Define the function to plot and save the size distribution chart
def plot_size_distribution(average_lengths, title_name='Size Distribution Chart'):
    # Define the bins for the histogram
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    # Remove NaN values and scale the lengths
    average_lengths = np.array(average_lengths) * 5
    average_lengths = average_lengths[~np.isnan(average_lengths)]
    # Calculate the number of elements in each bin
    counts, bin_edges = np.histogram(average_lengths, bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Fit a gamma distribution and calculate the PDF
    alpha_est, loc_est, beta_est = ss.gamma.fit(average_lengths, floc=0)  # Fix location to 0
    rv = ss.gamma(alpha_est, loc=loc_est, scale=beta_est)
    x = np.linspace(0, max(average_lengths), 1000)
    pdf = rv.pdf(x)

    # Scale the PDF to match the histogram's peak
    max_count = max(counts)
    pdf_scaled = pdf * max_count / max(pdf)

    # Plot the histogram and the PDF curve
    plt.hist(average_lengths, bins=bins, alpha=0.8, label='Particle Size Distribution', color='red', edgecolor='black', linewidth=1.5, hatch='//')
    plt.plot(x, pdf_scaled, color='blue', label='Distribution Curve', linewidth=2)

    # Calculate and display the mean size
    mean_size = np.mean(average_lengths)
    plt.axvline(mean_size, color='green', linestyle='--', linewidth=2, label=f'Mean Size: {mean_size:.2f}')

    # Set titles and labels
    plt.title(title_name)
    plt.xlabel('Size')
    plt.ylabel('Density')

    # Display the legend
    plt.legend()

    # Save the chart to an image file
    plt.savefig('size_distribution.png')
    plt.close()  # Close the plot to free up memory