from plotnine import *
import pandas as pd
import numpy as np 
from scipy.stats import spearmanr, chi2_contingency



def plot_target_correlation(df, target_col, feature_cols, title="Feature Correlation with Target"):
    """
    Plots a diverging bar chart showing the correlation of each feature 
    against a single target variable.
    """
    
    # 1. Calculate Correlations
    correlations = df[[target_col] + feature_cols].corr(method='spearman')[[target_col]]
    
    # Drop the target row itself
    correlations = correlations.drop(index=target_col).reset_index()
    correlations.columns = ['Feature', 'Correlation']
    
    # 2. Create Sorting Order
    correlations = correlations.sort_values('Correlation', ascending=True)
    
    # Force the sort order for plotting
    correlations['Feature'] = pd.Categorical(correlations['Feature'], categories=correlations['Feature'])

    # --- FIX IS HERE ---
    # 3. Create a pre-formatted label column
    correlations['corr_label'] = correlations['Correlation'].apply(lambda x: f"{x:.2f}")
    # -------------------

    # 4. Plot
    plot = (
        ggplot(correlations, aes(x='Feature', y='Correlation', fill='Correlation'))
        + geom_col(width=0.7)
        
        # Use the pre-calculated column 'corr_label'
        + geom_text(aes(label='corr_label'), size=9, nudge_y=0.02) 
        
        + scale_fill_gradient2(low="red", mid="white", high="blue", midpoint=0, limits=(-1, 1))
        + coord_flip() 
        + theme_minimal()
        + theme(
            figure_size=(8, 6),
            axis_title_y=element_blank(), 
            panel_grid_major_y=element_blank()
        )
        + labs(title=title, y="Spearman Correlation")
    )
    
    return plot


def plot_lower_triangle_heatmap(df, columns, title="Correlation Matrix", method='spearman'):
    """
    Generates a lower-triangle correlation heatmap using plotnine.
    """
    
    # 1. Calculate Correlation
    corr_matrix = df[columns].corr(method=method)

    # 2. Mask the upper triangle AND the diagonal
    # np.triu returns the upper triangle. k=0 includes diagonal, k=1 excludes it.
    # We want to HIDE the upper triangle, so we mask it.
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0) 
    
    # Apply mask (Values become NaN)
    corr_masked = corr_matrix.mask(mask)

    # 3. Melt to long format for plotnine
    corr_long = corr_masked.reset_index().melt(id_vars='index', var_name='Var2', value_name='Correlation')
    corr_long.rename(columns={'index': 'Var1'}, inplace=True)
    
    # Drop the NaN rows (the upper triangle) so they don't appear as empty gray boxes
    corr_long = corr_long.dropna()

    # 4. Formatting: Round numbers for the label
    corr_long['corr_label'] = corr_long['Correlation'].apply(lambda x: f"{x:.2f}")

    # 5. CRITICAL: Fix the sort order
    # Without this, ggplot sorts axes alphabetically, breaking the "triangle" shape.
    # We force the axes to follow the order of the 'columns' list provided.
    cat_type = pd.CategoricalDtype(categories=columns, ordered=True)
    corr_long['Var1'] = corr_long['Var1'].astype(cat_type)
    # For Var2, we reverse categories so the diagonal runs top-left to bottom-right neatly
    corr_long['Var2'] = corr_long['Var2'].astype(pd.CategoricalDtype(categories=reversed(columns), ordered=True))

    # 6. Build Plot
    heatmap = (
        ggplot(corr_long, aes(x='Var1', y='Var2', fill='Correlation'))
        + geom_tile(color="white") # Grid lines
        + geom_text(aes(label='corr_label'), size=9, color="black") # Add numbers
        + scale_fill_gradient2(low="blue", mid="white", high="red", midpoint=0, limits=(-1, 1))
        + coord_fixed() # Ensures tiles are square
        + theme_minimal()
        + theme(
            axis_text_x=element_text(rotation=45, hjust=1),
            axis_title=element_blank(), # Hides "Var1"/"Var2" labels
            panel_grid=element_blank(), # Removes background grid
            figure_size=(10, 8)
        )
        + labs(title=title)
    )
    
    return heatmap


def calculate_spearman_correlation(df, usage_col, barrier_col, label):
    # Create a local copy to avoid modifying the original dataframe
    temp_df = df[[usage_col, barrier_col]].copy()
    
    # Apply the 'Always' imputation logic (Always = No Barrier)
    temp_df.loc[temp_df[usage_col] == 'Always', barrier_col] = 0.0
    
    # Map usage to numeric ranks
    temp_df['Usage_Rank'] = temp_df[usage_col].map(usage_map)
    
    # Drop rows with missing values for a clean calculation
    temp_df = temp_df.dropna(subset=['Usage_Rank', barrier_col])
    
    # Calculate Spearman Correlation
    # Correlation between Usage (4 to 1) and Barrier (1 or 0)
    rho, p_value = spearmanr(temp_df['Usage_Rank'], temp_df[barrier_col])
    
    return {
        "Equipment": label,
        "Spearman_Rho (ρ)": round(rho, 4),
        "P-Value": round(p_value, 4),
        "N": len(temp_df),
        "Significant": "Yes" if p_value < 0.05 else "No"
    }

def calculate_binary_correlation(df, own_col, barrier_col, label):
    # 1. Clean data
    temp_df = df[[own_col, barrier_col]].dropna()
    
    # 2. Create contingency table and FORCE 2x2 shape
    # This ensures that even if '0' or '1' is missing from the data, the table has it
    ct = pd.crosstab(temp_df[own_col], temp_df[barrier_col])
    ct = ct.reindex(index=[0.0, 1.0], columns=[0.0, 1.0], fill_value=0)
    
    # 3. Check if we have enough data to actually run a Chi-Square
    # If a whole row or column is zero, correlation is technically undefined (0)
    if ct.sum().sum() == 0 or (ct.sum(axis=0) == 0).any() or (ct.sum(axis=1) == 0).any():
        return {
            "Analysis": label,
            "Phi_Coefficient": 0.0,
            "P-Value": 1.0,
            "N": len(temp_df),
            "Significant": "No (Insuff. Variance)"
        }

    # 4. Chi-square test
    chi2, p, dof, ex = chi2_contingency(ct)
    
    # 5. Calculate Phi Coefficient
    n = len(temp_df)
    phi = np.sqrt(chi2 / n)
    
    # 6. Directional Sign logic
    # % of Owners (1.0) citing barrier (1.0)
    owner_barrier_rate = ct.loc[1.0, 1.0] / ct.loc[1.0].sum() if ct.loc[1.0].sum() > 0 else 0
    # % of Non-owners (0.0) citing barrier (1.0)
    non_owner_barrier_rate = ct.loc[0.0, 1.0] / ct.loc[0.0].sum() if ct.loc[0.0].sum() > 0 else 0
    
    if owner_barrier_rate < non_owner_barrier_rate:
        phi = -phi

    return {
        "Analysis": label,
        "Phi_Coefficient": round(phi, 3),
        "P-Value": round(p, 3),
        "N": n,
        "Significant": "Yes" if p < 0.05 else "No"
    }