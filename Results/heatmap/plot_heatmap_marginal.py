#!/usr/bin/env python3
"""
Weekly emotion analysis with heatmap visualization.

Takes CSV data from weekly sentiment analysis and creates a nice heatmap
showing how emotions change over the semester weeks.

@Author: Jonatan Vider
@Date: 2025-09-22
@Version: 1.0.0
@Description: This script is used to plot the heatmap of the emotions and stress over the semester weeks.
@Usage: python plot_heatmap_marginal.py --csv /path/to/weekly_sentiment_metrics.csv --output /path/to/weekly_emotion_heatmap_marginal.png
@Example: python plot_heatmap_marginal.py --csv weekly_sentiment_metrics.csv --output weekly_emotion_heatmap_marginal.png
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec


def load_sentiment_data(csv_file):
    """Load the weekly sentiment CSV and extract what we need."""
    df = pd.read_csv(csv_file)
    return df


def extract_emotion_matrix(df):
    """Pull out emotion columns and make them into a matrix for plotting."""
    # Find all the emotion percentage columns
    emotion_cols = [
        col for col in df.columns if col.startswith("emotion_") and col.endswith("_pct")
    ]

    # Flip it so emotions are rows instead of columns
    emotion_data = df[emotion_cols].T

    # Clean up the names - remove the prefixes/suffixes
    emotion_names = [
        col.replace("emotion_", "").replace("_pct", "") for col in emotion_cols
    ]
    emotion_data.index = emotion_names
    emotion_data.columns = df["week_number"]

    return emotion_data, emotion_names


def sort_emotions_by_frequency(emotion_data):
    """Sort emotions so most common ones appear at the top of heatmap."""
    emotion_averages = emotion_data.mean(axis=1).sort_values(ascending=False)
    emotion_data_sorted = emotion_data.reindex(emotion_averages.index)
    emotion_names_sorted = emotion_averages.index.tolist()

    return emotion_data_sorted, emotion_names_sorted


def setup_figure_layout():
    """Set up the main figure with proper spacing and axes."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1, 4, 0.1],
        width_ratios=[4, 0.1],
        hspace=0.1,
        wspace=0.1,
    )

    # Create the different axes we need
    ax_main = fig.add_subplot(gs[1, 0])  # main heatmap
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)  # stress timeline
    ax_cbar = fig.add_subplot(gs[1, 1])  # colorbar

    return fig, ax_main, ax_top, ax_cbar


def create_main_heatmap(ax_main, emotion_data, emotion_names, df):
    """Draw the main emotion heatmap with all the percentage values."""
    # Create the actual heatmap
    im = ax_main.imshow(
        emotion_data.values, aspect="auto", cmap="viridis", interpolation="nearest"
    )

    # Set up tick labels
    ax_main.set_xticks(range(len(df)))
    ax_main.set_xticklabels(df["week_number"])
    ax_main.set_yticks(range(len(emotion_names)))
    ax_main.set_yticklabels(emotion_names)

    # Labels
    ax_main.set_xlabel("Week Number", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Emotion Type", fontsize=12, fontweight="bold")

    return im


def add_heatmap_annotations(ax_main, emotion_data, emotion_names, df):
    """Put the percentage numbers on each heatmap cell."""
    # Add white text on each cell
    for i in range(len(emotion_names)):
        for j in range(len(df)):
            value = emotion_data.iloc[i, j]
            if value > 0:  # only show when there's actually some percentage
                ax_main.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )


def create_stress_timeline(ax_top, df):
    """Create the stress progression plot at the top."""
    # Plot stress line with markers
    ax_top.plot(
        range(len(df)),
        df["avg_stress"],
        color="red",
        linewidth=3,
        marker="o",
        markersize=6,
    )
    # Fill underneath for visual effect
    ax_top.fill_between(range(len(df)), df["avg_stress"], alpha=0.3, color="red")

    # Styling
    ax_top.set_ylabel("Avg Stress", fontsize=11, fontweight="bold", color="red")
    ax_top.tick_params(axis="y", labelcolor="red")
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title(
        "Weekly Stress Progression", fontsize=12, fontweight="bold", pad=10
    )

    # Hide x labels since they're shared with main plot
    ax_top.tick_params(axis="x", labelbottom=False)


def add_titles_and_labels(fig, df):
    """Add the main title and footer information."""
    fig.suptitle(
        "Weekly Emotion & Stress Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )

    # Add context info at bottom
    start_date = df.iloc[0]["week_start"]
    end_date = df.iloc[-1]["week_end"]
    total_msgs = df["total_messages"].sum()

    fig.text(
        0.5,
        0.02,
        f"Semester Period: {start_date} to {end_date} | Total Messages: {total_msgs:,}",
        ha="center",
        fontsize=10,
        style="italic",
    )


def create_complete_visualization(df, emotion_data, emotion_names):
    """Put everything together into the final visualization."""
    # Set up the figure layout
    fig, ax_main, ax_top, ax_cbar = setup_figure_layout()

    # Create main heatmap
    im = create_main_heatmap(ax_main, emotion_data, emotion_names, df)

    # Add percentage annotations
    add_heatmap_annotations(ax_main, emotion_data, emotion_names, df)

    # Create stress timeline
    create_stress_timeline(ax_top, df)

    # Add colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label("Emotion Percentage (%)", fontsize=11, fontweight="bold")

    # Final titles and labels
    add_titles_and_labels(fig, df)

    plt.tight_layout()
    return fig


def print_analysis_summary(df, emotion_data):
    """Print some basic stats about what we found in the data."""
    print("\n--- Quick Analysis Summary ---")
    print(f"Analyzed {len(df)} weeks of data")
    print(f"Found {len(emotion_data)} different emotion types")

    # Find week with highest/lowest stress
    max_stress_week = df.loc[df["avg_stress"].idxmax(), "week_number"]
    min_stress_week = df.loc[df["avg_stress"].idxmin(), "week_number"]
    print(f"Most stressful week: Week {max_stress_week} ({df['avg_stress'].max():.3f})")
    print(
        f"Least stressful week: Week {min_stress_week} ({df['avg_stress'].min():.3f})"
    )

    # Show top emotions
    emotion_totals = emotion_data.mean(axis=1).sort_values(ascending=False)
    print("\nTop emotions throughout semester:")
    for i, (emotion, avg_pct) in enumerate(emotion_totals.head(3).items()):
        print(f"  {i+1}. {emotion}: {avg_pct:.1f}%")


def main():
    """Main function - loads data, creates visualization, saves it."""
    csv_file = "weekly_sentiment_metrics.csv"
    output_file = "weekly_emotion_heatmap_marginal.png"

    print("Loading sentiment data...")
    df = load_sentiment_data(csv_file)

    print("Processing emotion data...")
    emotion_data, emotion_names = extract_emotion_matrix(df)
    emotion_data, emotion_names = sort_emotions_by_frequency(emotion_data)

    print("Creating visualization...")
    fig = create_complete_visualization(df, emotion_data, emotion_names)

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved visualization as: {output_file}")

    # Show some insights
    print_analysis_summary(df, emotion_data)

    plt.show()


if __name__ == "__main__":
    main()
