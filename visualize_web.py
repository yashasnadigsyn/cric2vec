"""
Interactive 3D t-SNE Visualization using Plotly

Creates an interactive HTML visualization of player embeddings
that can be viewed in any web browser.
"""

from typing import Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def create_3d_visualization(
    embeddings: np.ndarray,
    player_names: List[str],
    labels: Optional[np.ndarray] = None,
    output_path: str = "plots/embeddings_3d.html",
    title: str = "Player Embeddings (3D t-SNE)",
    metadata: Optional[pd.DataFrame] = None
) -> None:
    """
    Create interactive 3D t-SNE visualization with Plotly.
    
    Args:
        embeddings: 2D array of shape (n_players, embedding_dim)
        player_names: List of player names
        labels: Optional cluster labels for coloring
        output_path: Path to save the HTML file
        title: Title for the visualization
        metadata: Optional DataFrame with player metadata for hover info
    """
    print(f"Running t-SNE (3D) on {len(embeddings)} embeddings...")
    
    # Run t-SNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'z': reduced[:, 2],
        'player': player_names,
    })
    
    if labels is not None:
        df['cluster'] = labels.astype(str)
    
    # Add metadata if provided
    if metadata is not None:
        for col in ['batting_style', 'bowling_style', 'position', 'country']:
            if col in metadata.columns:
                # Merge metadata
                meta_lookup = metadata.set_index('player_name')[col].to_dict()
                df[col] = [meta_lookup.get(name, 'Unknown') for name in player_names]
    
    # Create hover text
    hover_text = []
    for idx, row in df.iterrows():
        text = f"<b>{row['player']}</b>"
        if 'cluster' in df.columns:
            text += f"<br>Cluster: {row['cluster']}"
        if 'batting_style' in df.columns:
            text += f"<br>Batting: {row['batting_style']}"
        if 'bowling_style' in df.columns:
            text += f"<br>Bowling: {row['bowling_style']}"
        if 'country' in df.columns:
            text += f"<br>Country: {row['country']}"
        hover_text.append(text)
    
    df['hover'] = hover_text
    
    # Create 3D scatter plot
    if 'cluster' in df.columns:
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='cluster',
            hover_name='player',
            custom_data=['hover'],
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set1
        )
    else:
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            hover_name='player',
            custom_data=['hover'],
            title=title
        )
    
    # Update hover template
    fig.update_traces(
        hovertemplate='%{customdata[0]}<extra></extra>',
        marker=dict(size=5, opacity=0.7)
    )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='t-SNE Dim 1',
            yaxis_title='t-SNE Dim 2',
            zaxis_title='t-SNE Dim 3',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend_title_text='Cluster',
        font=dict(size=12)
    )
    
    # Save to HTML
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"Saved interactive 3D visualization to: {output_path}")


def create_clustered_3d_visualization(
    embeddings: np.ndarray,
    player_names: List[str],
    n_clusters: int = 6,
    output_path: str = "plots/embeddings_3d_clustered.html",
    title: str = "Player Embeddings with Clusters (3D)",
    metadata: Optional[pd.DataFrame] = None
) -> None:
    """
    Create 3D visualization with K-Means clustering.
    
    Args:
        embeddings: 2D array of embeddings
        player_names: List of player names
        n_clusters: Number of clusters
        output_path: Path to save HTML
        title: Plot title
        metadata: Optional player metadata
    """
    # Cluster
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Create visualization
    create_3d_visualization(
        embeddings=embeddings,
        player_names=player_names,
        labels=labels,
        output_path=output_path,
        title=title,
        metadata=metadata
    )


def main():
    """Generate 3D visualizations from the trained model."""
    from inference import CricInsights
    
    print("=" * 60)
    print("3D INTERACTIVE VISUALIZATION")
    print("=" * 60)
    
    # Load model and embeddings
    engine = CricInsights()
    
    # Get player names
    player_names = [engine.id_to_player[i] for i in range(engine.num_players)]
    
    # Load metadata if available
    metadata = None
    metadata_path = Path(__file__).parent / "player_metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        print(f"Loaded player metadata from {metadata_path}")
    
    # Create batter visualization
    print("\n--- Batter Embeddings (3D) ---")
    create_clustered_3d_visualization(
        embeddings=engine.bat_embeddings,
        player_names=player_names,
        n_clusters=6,
        output_path="plots/batters_3d.html",
        title="IPL Batter Embeddings (3D t-SNE with Clusters)",
        metadata=metadata
    )
    
    # Create bowler visualization
    print("\n--- Bowler Embeddings (3D) ---")
    create_clustered_3d_visualization(
        embeddings=engine.bowl_embeddings,
        player_names=player_names,
        n_clusters=6,
        output_path="plots/bowlers_3d.html",
        title="IPL Bowler Embeddings (3D t-SNE with Clusters)",
        metadata=metadata
    )
    
    print("\n" + "=" * 60)
    print("Done! Open the HTML files in a browser to explore:")
    print("  - plots/batters_3d.html")
    print("  - plots/bowlers_3d.html")
    print("=" * 60)


if __name__ == "__main__":
    main()
