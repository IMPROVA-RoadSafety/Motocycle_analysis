import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def usage_summary(df, col_index, values_not_always=[2,3], value_never=4, name=None):
    """
    Prints summary statistics for ownership and usage based on a survey column.

    Parameters:
    - df: pandas DataFrame
    - col_index: int, index of the column to analyze
    - values_not_always: list of int, values considered "own but not always using"
    - value_never: int, value considered "own but never using"
    - name: str, optional descriptive name of the item
    """
    if name is None:
        name = f"Column {col_index}"
    
    col = df.iloc[:, col_index]
    
    # Mean
    mean_value = col.mean()
    print(f"{name} - Mean:", round(mean_value, 2))
    
    # Own but not always using
    own_not_always = col[col.isin(values_not_always)].shape[0]
    pct_not_always = round(own_not_always / df.shape[0] * 100, 2)
    print(f"Own but not always using: {own_not_always}, Percentage: {pct_not_always}%")
    
    # Own but never using
    own_never = col[col == value_never].shape[0]
    pct_never = round(own_never / df.shape[0] * 100, 2)
    print(f"Own but never using: {own_never}, Percentage: {pct_never}%")
    
    # Return values in case you want to store them
    return {
        "mean": mean_value,
        "own_not_always": own_not_always,
        "pct_not_always": pct_not_always,
        "own_never": own_never,
        "pct_never": pct_never
    }


def analyze_ppe_usage(df, ppe_cols, demographics_cols, ppe_name="PPE"):
    """
    Filter for PPE owners, print ownership stats, and plot correlations.

    Parameters:
    - df: pandas DataFrame (full survey)
    - ppe_cols: dict mapping categories to column indices or lists
        Example: 
        {
            "ownership": 45,
            "satisfaction": [66, 67, 68, 69],
            "usage": 74,
            "reasons": [75, 76, 77],
            "statements": [88,89,90]
        }
    - demographics_cols: list of column names or indices for demographics
    - ppe_name: str, name of the PPE item for print statements
    """

    # --- 1. Filter owners ---
    ownership_col = ppe_cols["ownership"]
    owners_df = df[df.iloc[:, ownership_col] == 1]

    ownership_count = owners_df.shape[0]
    ownership_rate = round(ownership_count / df.shape[0] * 100, 2)

    print(f"{ppe_name} owners: {ownership_count}, Ownership rate: {ownership_rate}%")
    print(f"Ownership value counts:\n{df.iloc[:, ownership_col].value_counts()}")

    # --- 2. Flatten all other columns ---
    relevant_numbers = [
        n for key, val in ppe_cols.items() if key != "ownership"
        for n in (val if isinstance(val, list) else [val])
    ]
    relevant_columns = df.columns[relevant_numbers]

    # --- 3. Target column ---
    target_col = df.columns[ownership_col]

    # --- 4. Combine numeric columns for correlation ---
    numeric_cols = [target_col] + demographics_cols + relevant_columns.tolist()

    # --- 5. Plot correlations ---
    corr_fig = plot_target_correlations(
        df=df,
        target_col=target_col,
        numeric_cols=numeric_cols,
        label=f"{ppe_name} ownership"
    )

    return corr_fig, owners_df



def plot_target_correlations(
    df,
    target_col,
    numeric_cols,
    fillna_value=0,
    label=None,
    title_fontsize=20,
    tick_fontsize=16,
    annot_fontsize=14
):
    """
    Computes correlations of a target column with a set of numeric columns
    and plots a heatmap with fixed figure size 14x11.64, bigger text for presentations.

    Parameters:
    - df: pandas DataFrame
    - target_col: str, name of the target column
    - numeric_cols: list of str, columns to include (target + predictors)
    - fillna_value: value to fill NaNs before correlation (default=0)
    - label: str, optional name of the target variable for the title
    - title_fontsize: int, font size of the title
    - tick_fontsize: int, font size of the x and y tick labels
    - annot_fontsize: int, font size of the correlation annotations
    """
    # Subset the dataframe and fill NaNs
    df_subset = df[numeric_cols].fillna(fillna_value)

    # Compute correlations
    correlations = df_subset.corr()[target_col].drop(target_col)

    # Create fixed-size figure
    fig, ax = plt.subplots(figsize=(14, 11.64))

    # Plot heatmap
    sns.heatmap(
        correlations.to_frame().T,
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_fontsize},
        cmap="coolwarm",
        center=0,
        cbar=False,
        ax=ax
    )

    # Title
    if label:
        ax.set_title(f"Correlation with {label}", pad=20, fontsize=title_fontsize)
    else:
        ax.set_title(f"Correlation with {target_col}", pad=20, fontsize=title_fontsize)

    # Tick labels
    ax.tick_params(axis="x", rotation=45, labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    # Clean axes
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.despine()
    plt.tight_layout()
    plt.show()

    return fig



def plot_satisfaction_bar(
    df,
    cols,
    variables_df,
    title="Satisfaction by Characteristic",
    split_char="_",
    stacked=True,
    title_fontsize=20,
    tick_fontsize=16,
    xlabel_fontsize=18,
    ylabel_fontsize=18
):
    """
    Creates a bar chart of numeric satisfaction scores using seaborn/matplotlib,
    presentation-ready (fixed figure size, large text, no legend).

    Parameters:
    - df: pandas DataFrame with survey data
    - cols: list of column names to include
    - variables_df: DataFrame with columns 'Variable' and 'Etiqueta' for descriptive labels
    - title: plot title
    - split_char: character to shorten variable labels (optional)
    - stacked: bool, if True stacked bars, else grouped bars
    - title_fontsize: int, font size for the title
    - tick_fontsize: int, font size for x/y tick labels
    - xlabel_fontsize: int, font size for x-axis label
    - ylabel_fontsize: int, font size for y-axis label
    """

    # Melt to long format
    df_long = df[cols].melt(
        var_name="Characteristic",
        value_name="SatisfactionScore"
    ).dropna()

    # Replace variable names with descriptive labels
    label_map = dict(zip(variables_df["Variable"], variables_df["Etiqueta"]))
    df_long["Characteristic"] = df_long["Characteristic"].map(label_map).fillna(df_long["Characteristic"])

    # Shorten characteristic names
    df_long["Characteristic"] = df_long["Characteristic"].str.split(split_char).str[0]

    # Aggregate counts
    df_counts = df_long.groupby(["Characteristic", "SatisfactionScore"]).size().reset_index(name="Count")

    # Fixed figure size for presentation
    fig, ax = plt.subplots(figsize=(14, 11.64))

    if stacked:
        # Pivot for stacked bars
        pivot = df_counts.pivot(index="Characteristic", columns="SatisfactionScore", values="Count").fillna(0)
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", legend=False)
    else:
        sns.barplot(
            data=df_counts,
            x="Characteristic",
            y="Count",
            hue="SatisfactionScore",
            palette="viridis",
            ax=ax,
            dodge=True
        )
        ax.get_legend().remove()  # remove legend for grouped bars

    # Styling
    ax.set_title(title, fontsize=title_fontsize, pad=40)
    ax.set_xlabel("", fontsize=xlabel_fontsize)
    ax.set_ylabel("Count", fontsize=ylabel_fontsize)
    ax.tick_params(axis="x", rotation=45, labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    sns.despine()
    plt.tight_layout()
    plt.show()

    return fig



def plot_satisfaction_bar_plotly(
    df,
    cols,
    variables_df,
    title="Satisfaction by Characteristic",
    color_scale="Viridis",
    split_char="_",
    square=False,
    size=600
):
    """
    Creates a combined bar chart of numeric satisfaction scores using a continuous color scale
    for raw survey numeric data (e.g., 1-5 scales).
    """

    # Melt the dataframe to long format
    df_long = df[cols].melt(
        var_name="Characteristic",
        value_name="SatisfactionScore"
    ).dropna()

    # Replace variable names with labels
    df_long["Characteristic"] = df_long["Characteristic"].apply(
        lambda x: variable_to_label(x, variables_df)
    )

    # Shorten characteristic name
    df_long["Characteristic"] = (
        df_long["Characteristic"]
        .str.split(split_char)
        .str[0]
    )

    # Aggregate counts
    df_counts = (
        df_long
        .groupby(["Characteristic", "SatisfactionScore"])
        .size()
        .reset_index(name="Count")
    )

    # Plot
    fig = px.bar(
        df_counts,
        x="Characteristic",
        y="Count",
        color="SatisfactionScore",
        color_continuous_scale=color_scale,
        title=title
    )

    # Hide colorbar
    fig.update_layout(coloraxis_showscale=False)

    # Optional square layout
    if square:
        fig.update_layout(
            width=size,
            height=size
        )

    fig.show()
    return fig





def variable_to_label(variable, variables_df):
    """
    Given a variable name, return its descriptive label (Etiqueta).
    If not found, return the original variable.
    """
    label_map = dict(
        zip(variables_df["Variable"], variables_df["Etiqueta"])
    )
    return label_map.get(variable, variable)