"""
Visualization script for analyzing video embeddings from a VideoDataset pickle file.
Generates t-SNE and UMAP projections to understand latent space structure.

Usage:
    python test/visualize_embeddings.py --pickle path/to/dataset.pkl --output visualizations/
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_dataset(pickle_path: str):
    """Load VideoDataset from pickle file."""
    print(f"Loading dataset from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Loaded {len(dataset.video_datapoints)} videos")
    return dataset


def extract_embeddings(dataset, level='global', modalities=None) -> Dict[str, Dict]:
    """
    Extract embeddings from the dataset.
    
    Args:
        dataset: VideoDataset object
        level: 'global' for video-level or 'scene' for scene-level embeddings
        modalities: List of modalities to extract (e.g., ['video', 'audio', 'text'])
                   If None, extracts all available modalities
    
    Returns:
        Dictionary with structure: {modality: {'embeddings': array, 'labels': list, 'metadata': list}}
    """
    if modalities is None:
        modalities = ['video', 'audio', 'text']
    
    embeddings_data = {mod: {'embeddings': [], 'labels': [], 'metadata': []} for mod in modalities}
    
    for dp in dataset.video_datapoints:
        if level == 'global':
            # Extract global embeddings
            for modality in modalities:
                emb = dp.global_embeddings.get(modality)
                if emb is not None and isinstance(emb, torch.Tensor):
                    embeddings_data[modality]['embeddings'].append(emb.cpu().numpy())
                    embeddings_data[modality]['labels'].append(dp.video_name)
                    embeddings_data[modality]['metadata'].append({
                        'video_name': dp.video_name,
                        'num_scenes': len(dp.scenes),
                        'has_audio': getattr(dp, 'has_audio', True)
                    })
        
        elif level == 'scene':
            # Extract scene embeddings
            for scene_id, scene_data in dp.scene_embeddings.items():
                for modality in modalities:
                    emb = scene_data.get(modality)
                    if emb is not None and isinstance(emb, torch.Tensor):
                        embeddings_data[modality]['embeddings'].append(emb.cpu().numpy())
                        embeddings_data[modality]['labels'].append(f"{dp.video_name}_{scene_id}")
                        embeddings_data[modality]['metadata'].append({
                            'video_name': dp.video_name,
                            'scene_id': scene_id,
                            'has_audio': getattr(dp, 'has_audio', True)
                        })
    
    # Convert lists to numpy arrays
    for modality in modalities:
        if embeddings_data[modality]['embeddings']:
            embeddings_data[modality]['embeddings'] = np.vstack(embeddings_data[modality]['embeddings'])
            print(f"Extracted {len(embeddings_data[modality]['labels'])} {modality} embeddings "
                  f"({level} level) with shape {embeddings_data[modality]['embeddings'].shape}")
        else:
            print(f"Warning: No {modality} embeddings found at {level} level")
    
    return embeddings_data


def reduce_dimensions(embeddings: np.ndarray, method='tsne', n_components=2, 
                     space='feature', **kwargs) -> np.ndarray:
    """
    Reduce embedding dimensions using t-SNE, UMAP, or PCA.
    
    Args:
        embeddings: Input embeddings (n_samples, n_features)
        method: 'tsne', 'umap', or 'pca'
        n_components: Number of output dimensions (2 or 3)
        space: 'feature' to analyze embedding dimensions, 'sample' to analyze video clustering
        **kwargs: Additional parameters for the reduction method
    
    Returns:
        If space='feature': (n_features, n_components) - how embedding dims relate
        If space='sample': (n_samples, n_components) - how videos cluster
    """
    print(f"Reducing dimensions using {method.upper()} in {space} space...")
    print(f"Input shape: {embeddings.shape} (samples × features)")
    
    if space == 'feature':
        # Analyze how embedding dimensions relate to each other
        data = embeddings.T
        print(f"Analyzing feature space: {data.shape} (features × samples)")
    else:  # space == 'sample'
        # Analyze how videos/samples cluster
        data = embeddings
        print(f"Analyzing sample space: {data.shape} (samples × features)")
    
    if method == 'tsne':
        perplexity = kwargs.get('perplexity', min(30, len(data) - 1))
        reducer = TSNE(n_components=n_components, perplexity=perplexity, 
                      random_state=42, max_iter=1000, verbose=1)
    elif method == 'umap':
        n_neighbors = kwargs.get('n_neighbors', min(15, len(data) - 1))
        min_dist = kwargs.get('min_dist', 0.1)
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                           min_dist=min_dist, random_state=42, verbose=True)
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced = reducer.fit_transform(data)
    print(f"Reduced to shape {reduced.shape}")
    return reduced


def plot_2d_projection(reduced_emb: np.ndarray, labels: List[str], 
                       metadata: List[Dict], title: str, output_path: str,
                       color_by='video'):
    """
    Plot 2D projection of embedding feature dimensions.
    
    Args:
        reduced_emb: 2D reduced feature space (n_features, 2)
        labels: List of labels for each point (video names, not used for feature viz)
        metadata: List of metadata dicts
        title: Plot title
        output_path: Path to save the figure
        color_by: Not used for feature space visualization
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Visualize the embedding feature dimensions
    # Each point represents a dimension/feature in the embedding space
    n_features = reduced_emb.shape[0]
    
    scatter = ax.scatter(reduced_emb[:, 0], reduced_emb[:, 1], 
                        c=np.arange(n_features), cmap='viridis', 
                        alpha=0.6, s=30)
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(title + f'\n({n_features} embedding dimensions)', 
                fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, ax=ax, label='Feature Index')
    
    # Add text annotation
    ax.text(0.02, 0.98, f'Each point = 1 embedding dimension\nTotal: {n_features} dimensions',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_3d_projection(reduced_emb: np.ndarray, labels: List[str],
                       metadata: List[Dict], title: str, output_path: str,
                       color_by='video'):
    """Plot 3D projection of embedding feature dimensions."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Visualize the embedding feature dimensions
    n_features = reduced_emb.shape[0]
    colors = np.arange(n_features)
    
    scatter = ax.scatter(reduced_emb[:, 0], reduced_emb[:, 1], reduced_emb[:, 2],
                        c=colors, cmap='viridis', alpha=0.6, s=30)
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_zlabel('Component 3', fontsize=12)
    ax.set_title(title + f'\n({n_features} embedding dimensions)', 
                fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, ax=ax, label='Feature Index', shrink=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D plot to {output_path}")


def plot_modality_comparison(embeddings_data: Dict, method: str='tsne', output_dir: str="."):
    """
    Create dual comparison: feature space vs sample space for each modality.
    Top row: How embedding dimensions cluster (feature space)
    Bottom row: How videos cluster (sample space)
    """
    modalities = [m for m in embeddings_data.keys() if embeddings_data[m]['embeddings'].size > 0]
    
    if len(modalities) == 0:
        print("No embeddings to compare")
        return
    
    fig, axes = plt.subplots(2, len(modalities), figsize=(6*len(modalities), 10))
    if len(modalities) == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, modality in enumerate(modalities):
        emb = embeddings_data[modality]['embeddings']
        labels = embeddings_data[modality]['labels']
        metadata = embeddings_data[modality]['metadata']
        
        # Top row: FEATURE SPACE (dimensions)
        reduced_features = reduce_dimensions(emb, method=method, n_components=2, space='feature')
        n_features = reduced_features.shape[0]
        axes[0, idx].scatter(reduced_features[:, 0], reduced_features[:, 1], 
                            c=np.arange(n_features), cmap='viridis', alpha=0.6, s=20)
        axes[0, idx].set_title(f"{modality.capitalize()}\nFeature Space ({n_features} dims)", 
                              fontsize=11, fontweight='bold')
        axes[0, idx].set_xlabel('Component 1', fontsize=9)
        axes[0, idx].set_ylabel('Component 2', fontsize=9)
        
        # Bottom row: SAMPLE SPACE (videos)
        reduced_samples = reduce_dimensions(emb, method=method, n_components=2, space='sample')
        unique_videos = sorted(set(labels))
        color_map = {v: i for i, v in enumerate(unique_videos)}
        colors = [color_map[l] for l in labels]
        
        scatter = axes[1, idx].scatter(reduced_samples[:, 0], reduced_samples[:, 1], 
                                       c=colors, cmap='tab10' if len(unique_videos) <= 10 else 'tab20',
                                       alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        axes[1, idx].set_title(f"{modality.capitalize()}\nSample Space ({len(labels)} videos)", 
                              fontsize=11, fontweight='bold')
        axes[1, idx].set_xlabel('Component 1', fontsize=9)
        axes[1, idx].set_ylabel('Component 2', fontsize=9)
        
        # Add video labels
        for i, (x, y, label) in enumerate(zip(reduced_samples[:, 0], reduced_samples[:, 1], labels)):
            axes[1, idx].annotate(label[:15], (x, y), fontsize=6, alpha=0.7)
    
    plt.suptitle(f"Dual Perspective: Feature vs Sample Space ({method.upper()})", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"dual_perspective_{method}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dual perspective comparison to {output_path}")


def plot_embedding_statistics(embeddings_data: Dict, output_dir: str):
    """
    Plot statistics about embeddings: dimensionality, norms, variance, etc.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    modalities = [m for m in embeddings_data.keys() if embeddings_data[m]['embeddings'].size > 0]
    
    # Plot 1: Embedding dimensions
    dims = [embeddings_data[m]['embeddings'].shape[1] for m in modalities]
    counts = [embeddings_data[m]['embeddings'].shape[0] for m in modalities]
    
    axes[0, 0].bar(modalities, dims, color='skyblue')
    axes[0, 0].set_title('Embedding Dimensions by Modality', fontweight='bold')
    axes[0, 0].set_ylabel('Dimension')
    axes[0, 0].set_xlabel('Modality')
    
    # Add count labels
    for i, (mod, dim, count) in enumerate(zip(modalities, dims, counts)):
        axes[0, 0].text(i, dim + 10, f"n={count}", ha='center', fontsize=9)
    
    # Plot 2: L2 norms distribution
    use_kde = False
    for modality in modalities:
        emb = embeddings_data[modality]['embeddings']
        norms = np.linalg.norm(emb, axis=1)
        
        # Check if norms have sufficient range for histogram
        norm_range = norms.max() - norms.min()
        if norm_range < 1e-6:  # Too little variance for histogram
            use_kde = True
        
        if not use_kde:
            try:
                axes[0, 1].hist(norms, alpha=0.5, bins='auto', label=modality)
            except ValueError:
                # Fall back to KDE if histogram fails
                use_kde = True
    
    if use_kde:
        # Clear the axis and use KDE instead
        axes[0, 1].clear()
        from scipy import stats
        for modality in modalities:
            emb = embeddings_data[modality]['embeddings']
            norms = np.linalg.norm(emb, axis=1)
            
            # Use KDE for smooth density estimation
            if len(norms) > 1:
                kde = stats.gaussian_kde(norms)
                x_range = np.linspace(norms.min() - 0.1, norms.max() + 0.1, 100)
                axes[0, 1].plot(x_range, kde(x_range), alpha=0.7, label=modality, linewidth=2)
            else:
                # Single point - just mark it
                axes[0, 1].axvline(norms[0], alpha=0.7, label=modality, linewidth=2)
        axes[0, 1].set_ylabel('Density')
    
    axes[0, 1].set_title('L2 Norm Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('L2 Norm')
    if not use_kde:
        axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Plot 3: Mean embedding per dimension (first modality)
    if modalities:
        main_modality = modalities[0]
        emb = embeddings_data[main_modality]['embeddings']
        mean_per_dim = emb.mean(axis=0)
        std_per_dim = emb.std(axis=0)
        
        dims_to_show = min(50, len(mean_per_dim))
        x = np.arange(dims_to_show)
        axes[1, 0].plot(x, mean_per_dim[:dims_to_show], label='Mean', color='blue')
        axes[1, 0].fill_between(x, 
                                mean_per_dim[:dims_to_show] - std_per_dim[:dims_to_show],
                                mean_per_dim[:dims_to_show] + std_per_dim[:dims_to_show],
                                alpha=0.3, color='blue')
        axes[1, 0].set_title(f'{main_modality.capitalize()} Embedding Statistics (first {dims_to_show} dims)', 
                            fontweight='bold')
        axes[1, 0].set_xlabel('Dimension Index')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Variance explained (PCA)
    if modalities:
        main_modality = modalities[0]
        emb = embeddings_data[main_modality]['embeddings']
        
        n_components = min(50, emb.shape[0], emb.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(emb)
        
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        axes[1, 1].plot(cumsum, marker='o', markersize=3)
        axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        axes[1, 1].set_title(f'{main_modality.capitalize()} PCA Variance Explained', fontweight='bold')
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('Cumulative Variance Explained')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'embedding_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved statistics to {output_path}")


def plot_inter_modality_similarity(embeddings_data: Dict, output_dir: str):
    """
    Compute and visualize similarity between different modalities (same samples).
    Since modalities have different dimensions, we compute sample-to-sample 
    similarity separately and create a correlation matrix.
    """
    modalities = [m for m in embeddings_data.keys() if embeddings_data[m]['embeddings'].size > 0]
    
    if len(modalities) < 2:
        print("Need at least 2 modalities for inter-modality analysis")
        return
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Compute pairwise similarities for all modality pairs
    n_modalities = len(modalities)
    similarity_matrices = {}
    
    for i in range(n_modalities):
        for j in range(i + 1, n_modalities):
            mod1, mod2 = modalities[i], modalities[j]
            
            # Get embeddings for matching samples
            emb1 = embeddings_data[mod1]['embeddings']
            emb2 = embeddings_data[mod2]['embeddings']
            labels1 = embeddings_data[mod1]['labels']
            labels2 = embeddings_data[mod2]['labels']
            
            # Find matching samples by label
            common_labels = set(labels1) & set(labels2)
            if not common_labels:
                print(f"Warning: No common samples between {mod1} and {mod2}")
                continue
            
            # Extract matching embeddings
            idx1 = [i for i, l in enumerate(labels1) if l in common_labels]
            idx2 = [i for i, l in enumerate(labels2) if l in common_labels]
            
            # Sort to ensure same order
            sorted_labels = sorted(common_labels)
            label_to_idx1 = {l: i for i, l in enumerate(labels1)}
            label_to_idx2 = {l: i for i, l in enumerate(labels2)}
            
            idx1 = [label_to_idx1[l] for l in sorted_labels]
            idx2 = [label_to_idx2[l] for l in sorted_labels]
            
            emb1_matched = emb1[idx1]
            emb2_matched = emb2[idx2]
            
            # Check if embeddings have same dimension
            if emb1_matched.shape[1] != emb2_matched.shape[1]:
                print(f"Skipping {mod1}-{mod2}: incompatible dimensions ({emb1_matched.shape[1]} vs {emb2_matched.shape[1]})")
                continue
            
            # Normalize
            emb1_norm = emb1_matched / (np.linalg.norm(emb1_matched, axis=1, keepdims=True) + 1e-8)
            emb2_norm = emb2_matched / (np.linalg.norm(emb2_matched, axis=1, keepdims=True) + 1e-8)
            
            # Compute diagonal similarity (same sample across modalities)
            sample_similarities = np.sum(emb1_norm * emb2_norm, axis=1)
            
            similarity_matrices[f"{mod1}-{mod2}"] = {
                'mean': sample_similarities.mean(),
                'std': sample_similarities.std(),
                'values': sample_similarities,
                'labels': sorted_labels
            }
    
    if not similarity_matrices:
        print("No compatible modality pairs found for inter-modality analysis")
        return
    
    # Create visualization
    n_pairs = len(similarity_matrices)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]
    
    for idx, (pair_name, sim_data) in enumerate(similarity_matrices.items()):
        # Plot bar chart of sample-wise similarities
        labels = sim_data['labels']
        values = sim_data['values']
        
        x = np.arange(len(labels))
        axes[idx].bar(x, values, color='steelblue', alpha=0.7)
        axes[idx].axhline(y=sim_data['mean'], color='r', linestyle='--', 
                         label=f"Mean: {sim_data['mean']:.3f}")
        axes[idx].set_title(f"{pair_name}\nCosine Similarity", fontweight='bold')
        axes[idx].set_xlabel('Sample Index')
        axes[idx].set_ylabel('Cosine Similarity')
        axes[idx].set_ylim([-1, 1])
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add text with statistics
        textstr = f'μ={sim_data["mean"]:.3f}\nσ={sim_data["std"]:.3f}'
        axes[idx].text(0.02, 0.98, textstr, transform=axes[idx].transAxes,
                      verticalalignment='top', bbox=dict(boxstyle='round', 
                      facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'inter_modality_similarity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved inter-modality similarity to {output_path}")


def plot_cross_modal_alignment(embeddings_data: Dict, method: str='tsne', output_dir: str="."):
    """
    Visualize cross-modal alignment in a SHARED embedding space.
    
    All modalities are projected into the same 2D space using the same 
    dimensionality reduction, so we can see:
    - Do different modalities form separate clusters or overlap?
    - Is the same video's embedding close across modalities?
    
    Since modalities have different dimensions, we first normalize each modality's
    embeddings, then apply PCA to bring them to a common dimension before t-SNE/UMAP.
    """
    modalities = [m for m in embeddings_data.keys() if embeddings_data[m]['embeddings'].size > 0]
    
    if len(modalities) < 2:
        print("Need at least 2 modalities for cross-modal alignment")
        return
    
    # Find common videos across all modalities
    common_labels = set(embeddings_data[modalities[0]]['labels'])
    for modality in modalities[1:]:
        common_labels &= set(embeddings_data[modality]['labels'])
    
    if not common_labels:
        print("No common videos across modalities")
        return
    
    common_labels = sorted(common_labels)
    print(f"Found {len(common_labels)} common videos across {len(modalities)} modalities")
    
    # Step 1: Normalize and project each modality to common dimension using PCA
    common_dim = min(50, min(embeddings_data[mod]['embeddings'].shape[1] for mod in modalities))
    
    all_embeddings = []
    all_labels = []
    all_modalities = []
    
    for modality in modalities:
        emb = embeddings_data[modality]['embeddings']
        labels = embeddings_data[modality]['labels']
        
        # Filter to common videos only
        indices = [i for i, l in enumerate(labels) if l in common_labels]
        emb_common = emb[indices]
        labels_common = [labels[i] for i in indices]
        
        # Normalize
        emb_normalized = emb_common / (np.linalg.norm(emb_common, axis=1, keepdims=True) + 1e-8)
        
        # Project to common dimension with PCA
        pca = PCA(n_components=min(common_dim, emb_normalized.shape[0], emb_normalized.shape[1]))
        emb_pca = pca.fit_transform(emb_normalized)
        
        # Pad to common_dim if necessary
        if emb_pca.shape[1] < common_dim:
            padding = np.zeros((emb_pca.shape[0], common_dim - emb_pca.shape[1]))
            emb_pca = np.hstack([emb_pca, padding])
        
        all_embeddings.append(emb_pca)
        all_labels.extend(labels_common)
        all_modalities.extend([modality] * len(labels_common))
    
    # Step 2: Concatenate all embeddings and reduce to 2D in SHARED space
    combined_embeddings = np.vstack(all_embeddings)
    print(f"Combined embeddings shape: {combined_embeddings.shape}")
    
    reduced = reduce_dimensions(combined_embeddings, method=method, n_components=2, space='sample')
    
    # Step 3: Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by video, marker/shape by modality
    video_colors = {v: i for i, v in enumerate(common_labels)}
    modality_markers = {'text': 'o', 'video': 's', 'audio': '^'}
    modality_sizes = {'text': 150, 'video': 200, 'audio': 150}
    
    colors_palette = plt.cm.tab10 if len(common_labels) <= 10 else plt.cm.tab20
    
    # Plot each modality with different marker
    for modality in modalities:
        mask = [m == modality for m in all_modalities]
        x = reduced[mask, 0]
        y = reduced[mask, 1]
        colors = [video_colors[all_labels[i]] for i in range(len(all_labels)) if all_modalities[i] == modality]
        
        ax.scatter(x, y, 
                  c=colors, cmap=colors_palette,
                  marker=modality_markers.get(modality, 'o'),
                  s=modality_sizes.get(modality, 150),
                  alpha=0.7, 
                  edgecolors='black', 
                  linewidth=1.5,
                  label=modality.capitalize())
    
    # Draw lines connecting same video across modalities
    for video in common_labels:
        points = []
        for i, (label, mod) in enumerate(zip(all_labels, all_modalities)):
            if label == video:
                points.append((reduced[i, 0], reduced[i, 1]))
        
        if len(points) > 1:
            # Draw polygon connecting all modalities for this video
            points_array = np.array(points)
            color_idx = video_colors[video]
            color = colors_palette(color_idx / max(len(common_labels) - 1, 1))
            
            # Draw lines between consecutive points
            for j in range(len(points)):
                for k in range(j + 1, len(points)):
                    ax.plot([points[j][0], points[k][0]], 
                           [points[j][1], points[k][1]], 
                           'k-', alpha=0.2, linewidth=0.5, zorder=0)
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(f'Cross-Modal Alignment in Shared Space ({method.upper()})\n' + 
                f'Same color = same video, Different shapes = different modalities',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add text annotation
    textstr = f'{len(common_labels)} videos × {len(modalities)} modalities\n' + \
              'Lines connect same video across modalities'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.7), fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"cross_modal_shared_space_{method}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-modal shared space plot to {output_path}")
    
    # Also create distance analysis
    plot_cross_modal_distances(embeddings_data, common_labels, output_dir)


def plot_cross_modal_distances(embeddings_data: Dict, common_labels: List[str], output_dir: str):
    """
    Compute and visualize the distance between the same video's embeddings 
    across different modalities (in the original embedding space).
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    modalities = [m for m in embeddings_data.keys() if embeddings_data[m]['embeddings'].size > 0]
    
    if len(modalities) < 2:
        return
    
    # Compute cosine similarity for each video across modality pairs
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(common_labels))
    width = 0.8 / (len(modalities) * (len(modalities) - 1) // 2)
    offset = 0
    
    for i in range(len(modalities)):
        for j in range(i + 1, len(modalities)):
            mod1, mod2 = modalities[i], modalities[j]
            
            emb1 = embeddings_data[mod1]['embeddings']
            emb2 = embeddings_data[mod2]['embeddings']
            labels1 = embeddings_data[mod1]['labels']
            labels2 = embeddings_data[mod2]['labels']
            
            # Check if same dimensions
            if emb1.shape[1] != emb2.shape[1]:
                continue
            
            similarities = []
            for label in common_labels:
                idx1 = labels1.index(label)
                idx2 = labels2.index(label)
                
                # Cosine similarity
                sim = cosine_similarity(emb1[idx1:idx1+1], emb2[idx2:idx2+1])[0, 0]
                similarities.append(sim)
            
            ax.bar(x_pos + offset, similarities, width, 
                  label=f"{mod1}-{mod2}", alpha=0.7)
            offset += width
    
    ax.set_xlabel('Video', fontsize=11)
    ax.set_ylabel('Cosine Similarity', fontsize=11)
    ax.set_title('Cross-Modal Embedding Similarity\n(Same video across modalities)', 
                fontweight='bold', fontsize=12)
    ax.set_xticks(x_pos + 0.4)
    ax.set_xticklabels([l[:15] for l in common_labels], rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cross_modal_similarity_bars.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-modal similarity bars to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize embeddings from VideoDataset')
    parser.add_argument('--pickle', type=str, required=True, help='Path to VideoDataset pickle file')
    parser.add_argument('--output', type=str, default='visualizations/', help='Output directory for plots')
    parser.add_argument('--level', type=str, choices=['global', 'scene'], default='global',
                       help='Embedding level to visualize')
    parser.add_argument('--modalities', type=str, nargs='+', default=['video', 'audio', 'text'],
                       help='Modalities to visualize')
    parser.add_argument('--methods', type=str, nargs='+', default=['tsne', 'umap', 'pca'],
                       help='Dimensionality reduction methods')
    parser.add_argument('--3d', action='store_true', help='Generate 3D visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(args.pickle)
    
    # Extract embeddings
    embeddings_data = extract_embeddings(dataset, level=args.level, modalities=args.modalities)
    
    # Plot statistics
    print("\nGenerating embedding statistics...")
    plot_embedding_statistics(embeddings_data, args.output)
    
    # Plot inter-modality similarity
    print("\nGenerating inter-modality similarity plots...")
    plot_inter_modality_similarity(embeddings_data, args.output)
    
    # Generate visualizations for each modality and method
    for modality in args.modalities:
        if not embeddings_data[modality]['embeddings'].size:
            print(f"Skipping {modality} - no embeddings found")
            continue
        
        emb = embeddings_data[modality]['embeddings']
        labels = embeddings_data[modality]['labels']
        metadata = embeddings_data[modality]['metadata']
        
        for method in args.methods:
            print(f"\nProcessing {modality} with {method}...")
            
            # 2D visualization
            reduced_2d = reduce_dimensions(emb, method=method, n_components=2, space='feature')
            output_path = os.path.join(args.output, f"{modality}_{method}_2d_{args.level}.png")
            plot_2d_projection(reduced_2d, labels, metadata,
                             f"{modality.capitalize()} Embeddings - {method.upper()} ({args.level} level)",
                             output_path, color_by='video')
            
            # 3D visualization
            if args.__dict__['3d']:
                reduced_3d = reduce_dimensions(emb, method=method, n_components=3, space='feature')
                output_path = os.path.join(args.output, f"{modality}_{method}_3d_{args.level}.png")
                plot_3d_projection(reduced_3d, labels, metadata,
                                 f"{modality.capitalize()} Embeddings - {method.upper()} 3D ({args.level} level)",
                                 output_path, color_by='video')
    
    # Modality comparison (dual perspective)
    for method in args.methods:
        print(f"\nGenerating dual perspective comparison with {method}...")
        plot_modality_comparison(embeddings_data, method=method, output_dir=args.output)
    
    # Cross-modal sample alignment
    for method in args.methods:
        print(f"\nGenerating cross-modal alignment analysis with {method}...")
        plot_cross_modal_alignment(embeddings_data, method=method, output_dir=args.output)
    
    print(f"\nAll visualizations saved to {args.output}")
    print("\nSummary of generated files:")
    for file in sorted(os.listdir(args.output)):
        if file.endswith('.png'):
            print(f"  - {file}")


if __name__ == "__main__":
    main()
