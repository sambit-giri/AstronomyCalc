import matplotlib.pyplot as plt
import numpy as np

def plot_universe_pie(cosmo_dict, title="Content of the Universe"):
    """
    Create a pie chart showing the content of the universe based on input cosmological parameters.
    
    Parameters:
        cosmo_dict (dict): A dictionary containing the fractions of components in the universe.
                           Example: {'Om': 0.31, 'Or': 0.0, 'Ok': 0.0, 'Ode': 0.69, 'h': 0.67}
                           Note: Ensure the fractions sum approximately to 1 (excluding h).
    """
    # Prepare data
    labels = {
        'Om': 'Matter',
        'Or': 'Radiation',
        'Ok': 'Curvature',
        'Ode': 'Dark Energy'
    }
    
    # Filter out the 'h' key or any non-density keys
    density_labels = {key: labels.get(key, key) for key in cosmo_dict if key != 'h'}
    densities = {k: v for k, v in cosmo_dict.items() if k != 'h' and v > 0}

    # Extract labels and values
    pie_labels = [density_labels[k] for k in densities]
    pie_values = list(densities.values())
    
    # Create pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        pie_values,
        labels=pie_labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Paired.colors
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()