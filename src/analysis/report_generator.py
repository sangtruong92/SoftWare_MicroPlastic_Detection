"""
Report Generator - HTML visualization reports
Creates interactive HTML reports with charts for benchmark results
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import base64
from io import BytesIO

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ReportGenerator:
    """Generate HTML reports with visualizations"""
    
    def __init__(self):
        """Initialize report generator"""
        self.template = self._load_template()
    
    def generate_benchmark_report(self, benchmark_results: Dict[str, Any], 
                                  output_path: str) -> str:
        """
        Generate comprehensive HTML benchmark report.
        
        Args:
            benchmark_results: Dictionary containing benchmark metrics
            output_path: Path to save HTML report
            
        Returns:
            Path to generated HTML file
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for report generation")
        
        # Extract data
        quick_metrics = benchmark_results.get('quick_analysis', {})
        deep_metrics = benchmark_results.get('deep_analysis', {})
        watershed_metrics = benchmark_results.get('watershed_analysis', {})
        ml_metrics = benchmark_results.get('ml_benchmark', {})
        
        # Generate charts
        charts = {}
        
        # 1. Precision/Recall Comparison
        charts['precision_recall'] = self._create_precision_recall_chart(
            quick_metrics, deep_metrics, watershed_metrics, ml_metrics
        )
        
        # Extract ground truth distributions if available
        gt_distributions = benchmark_results.get('ground_truth_distributions', {})
        gt_shapes = gt_distributions.get('shape_distribution', {}) if gt_distributions else {}
        gt_colors = gt_distributions.get('color_distribution', {}) if gt_distributions else {}
        gt_areas = gt_distributions.get('area_distribution', []) if gt_distributions else []
        
        # 2. Shape Distribution Comparison (Quick vs Deep vs ML vs Ground Truth)
        if 'shape_distribution' in quick_metrics and 'shape_distribution' in deep_metrics:
            ml_shapes = ml_metrics.get('shape_distribution', {}) if ml_metrics else {}
            charts['shape_comparison'] = self._create_shape_comparison_chart(
                quick_metrics['shape_distribution'], 
                deep_metrics['shape_distribution'],
                ml_shapes,
                gt_shapes
            )
        
        # 3. Color Distribution Comparison (Quick vs Deep vs ML vs Ground Truth)
        if 'color_distribution' in quick_metrics and 'color_distribution' in deep_metrics:
            ml_colors = ml_metrics.get('color_distribution', {}) if ml_metrics else {}
            charts['color_comparison'] = self._create_color_comparison_chart(
                quick_metrics['color_distribution'], 
                deep_metrics['color_distribution'],
                ml_colors,
                gt_colors
            )
        
        # 4. Area Distribution Comparison (Quick vs Deep vs ML vs Ground Truth)
        if 'area_distribution' in quick_metrics and 'area_distribution' in deep_metrics:
            ml_areas = ml_metrics.get('area_distribution', []) if ml_metrics else []
            charts['area_comparison'] = self._create_area_comparison_chart(
                quick_metrics['area_distribution'],
                deep_metrics['area_distribution'],
                ml_areas,
                gt_areas
            )
        
        # Legacy charts for backward compatibility
        if 'area_distribution' in deep_metrics and not charts.get('area_comparison'):
            charts['area_dist'] = self._create_area_distribution_chart(
                deep_metrics['area_distribution']
            )
        
        if 'shape_distribution' in deep_metrics and not charts.get('shape_comparison'):
            charts['shape_dist'] = self._create_shape_distribution_chart(
                deep_metrics['shape_distribution']
            )
        
        if 'color_distribution' in deep_metrics and not charts.get('color_comparison'):
            charts['color_dist'] = self._create_color_distribution_chart(
                deep_metrics['color_distribution']
            )
        
        # 5. Confusion Matrix (if available)
        if 'confusion_matrix' in deep_metrics:
            charts['confusion'] = self._create_confusion_matrix_chart(
                deep_metrics['confusion_matrix']
            )
        
        # Build HTML content
        html_content = self._build_html(benchmark_results, charts)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _create_precision_recall_chart(self, quick: Dict, deep: Dict, 
                                      watershed: Dict, ml: Dict = None) -> str:
        """Create precision/recall comparison bar chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        methods = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for name, metrics in [('Quick', quick), ('Deep', deep), ('Watershed', watershed), ('ML Benchmark', ml)]:
            if metrics and 'precision' in metrics:
                methods.append(name)
                precisions.append(metrics.get('precision', 0))
                recalls.append(metrics.get('recall', 0))
                f1_scores.append(metrics.get('f1_score', 0))
        
        if not methods:
            return ""
        
        x = np.arange(len(methods))
        width = 0.25
        
        # Precision and Recall
        ax1.bar(x - width, precisions, width, label='Precision', color='#4CAF50', alpha=0.8)
        ax1.bar(x, recalls, width, label='Recall', color='#2196F3', alpha=0.8)
        ax1.bar(x + width, f1_scores, width, label='F1-Score', color='#FF9800', alpha=0.8)
        
        ax1.set_xlabel('Analysis Method', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Precision, Recall, and F1-Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precisions, recalls, f1_scores)):
            ax1.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=9)
            ax1.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=9)
            ax1.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Detection counts
        detected = [metrics.get('detected', 0) for metrics in [quick, deep, watershed, ml] if metrics]
        ground_truth = deep.get('ground_truth', 0) if deep else 0
        
        if detected:
            ax2.bar(methods, detected, color='#9C27B0', alpha=0.8, label='Detected')
            ax2.axhline(y=ground_truth, color='red', linestyle='--', linewidth=2, 
                       label=f'Ground Truth: {ground_truth}')
            ax2.set_xlabel('Analysis Method', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Particle Count', fontsize=12, fontweight='bold')
            ax2.set_title('Detection Count Comparison', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, val in enumerate(detected):
                ax2.text(i, val + max(detected) * 0.02, str(val), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_shape_comparison_chart(self, quick_shapes: Dict, deep_shapes: Dict, ml_shapes: Dict = None, gt_shapes: Dict = None) -> str:
        """Create shape distribution comparison chart for Quick vs Deep vs ML vs Ground Truth"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if ml_shapes is None:
            ml_shapes = {}
        if gt_shapes is None:
            gt_shapes = {}
        
        # Get all unique shapes
        all_shapes_set = set(list(quick_shapes.keys()) + list(deep_shapes.keys()) + list(ml_shapes.keys()) + list(gt_shapes.keys()))
        
        # Define logical grouping order - related categories placed together
        preferred_order = [
            'Fiber', 'Fiber/Filament', 'Filament',  # Fiber group
            'Fragment',  # Fragment
            'Irregular',  # Irregular
            'Microbead', 'Microbead/Pellet', 'Pellet'  # Bead group
        ]
        
        # Place shapes in preferred order, then add any remaining shapes
        all_shapes = [s for s in preferred_order if s in all_shapes_set]
        all_shapes += sorted([s for s in all_shapes_set if s not in preferred_order])
        
        if not all_shapes:
            return ""
        
        x = np.arange(len(all_shapes))
        width = 0.2
        
        quick_counts = [quick_shapes.get(shape, 0) for shape in all_shapes]
        deep_counts = [deep_shapes.get(shape, 0) for shape in all_shapes]
        gt_counts = [gt_shapes.get(shape, 0) for shape in all_shapes]
        ml_counts = [ml_shapes.get(shape, 0) for shape in all_shapes]
        
        offset = -1.5 * width
        if gt_shapes:
            ax.bar(x + offset, gt_counts, width, label='Ground Truth', color='#FF5722', alpha=0.8, edgecolor='black', hatch='//')
        offset += width
        ax.bar(x + offset, quick_counts, width, label='Quick Analysis', color='#2196F3', alpha=0.8, edgecolor='black')
        offset += width
        ax.bar(x + offset, deep_counts, width, label='Deep Analysis', color='#4CAF50', alpha=0.8, edgecolor='black')
        offset += width
        if ml_shapes:
            ax.bar(x + offset, ml_counts, width, label='ML Benchmark', color='#9C27B0', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Shape Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Shape Distribution: Quick vs Deep vs ML vs Ground Truth', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_shapes, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        all_counts = gt_counts + quick_counts + deep_counts + ml_counts
        max_count = max(all_counts) if all_counts else 1
        offset = -1.5 * width
        for i, (gc, qc, dc, mc) in enumerate(zip(gt_counts, quick_counts, deep_counts, ml_counts)):
            if gc > 0 and gt_shapes:
                ax.text(i + offset, gc + max_count * 0.02, 
                       str(gc), ha='center', va='bottom', fontsize=8)
            offset_tmp = offset + width
            if qc > 0:
                ax.text(i + offset_tmp, qc + max_count * 0.02, 
                       str(qc), ha='center', va='bottom', fontsize=8)
            offset_tmp += width
            if dc > 0:
                ax.text(i + offset_tmp, dc + max_count * 0.02, 
                       str(dc), ha='center', va='bottom', fontsize=8)
            offset_tmp += width
            if mc > 0 and ml_shapes:
                ax.text(i + offset_tmp, mc + max_count * 0.02, 
                       str(mc), ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_color_comparison_chart(self, quick_colors: Dict, deep_colors: Dict, ml_colors: Dict = None, gt_colors: Dict = None) -> str:
        """Create color distribution comparison chart for Quick vs Deep vs ML vs Ground Truth"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if ml_colors is None:
            ml_colors = {}
        if gt_colors is None:
            gt_colors = {}
        
        # Get all unique colors
        all_colors_set = set(list(quick_colors.keys()) + list(deep_colors.keys()) + list(ml_colors.keys()) + list(gt_colors.keys()))
        
        # Define logical grouping order - 4 main fluorescent colors first
        preferred_order = [
            'Blue',    # Blue first
            'Green',   # Green second
            'Red',     # Red third
            'Yellow'   # Yellow fourth
        ]
        
        # Place colors in preferred order, then add any remaining colors
        all_colors = [c for c in preferred_order if c in all_colors_set]
        all_colors += sorted([c for c in all_colors_set if c not in preferred_order])
        
        if not all_colors:
            return ""
        
        # Fluorescent color mapping - matches synthetic generation
        visual_colors = {
            'Red': '#FF0000', 'Green': '#00FF00', 'Blue': '#0000FF', 'Yellow': '#FFFF00',
            'Orange': '#FFA500', 'Purple': '#800080', 'Pink': '#FFC0CB', 'Brown': '#A52A2A',
            'Black': '#000000', 'White': '#FFFFFF', 'Gray': '#808080', 'Cyan': '#00FFFF'
        }
        
        x = np.arange(len(all_colors))
        width = 0.2
        
        quick_counts = [quick_colors.get(color, 0) for color in all_colors]
        deep_counts = [deep_colors.get(color, 0) for color in all_colors]
        gt_counts = [gt_colors.get(color, 0) for color in all_colors]
        ml_counts = [ml_colors.get(color, 0) for color in all_colors]
        
        # Create bars with actual fluorescent colors, using alpha to distinguish methods
        offset = -1.5 * width
        for i, color_name in enumerate(all_colors):
            base_color = visual_colors.get(color_name, '#808080')
            # Ground Truth - full saturation with hatching
            if gt_colors:
                ax.bar(i + offset, gt_counts[i], width,
                      color=base_color, alpha=1.0, edgecolor='black', linewidth=2, hatch='//')
            offset_val = offset + width
            # Quick Analysis - lighter shade
            ax.bar(i + offset_val, quick_counts[i], width, 
                  color=base_color, alpha=0.4, edgecolor='black', linewidth=1)
            offset_val += width
            # Deep Analysis - medium shade
            ax.bar(i + offset_val, deep_counts[i], width,
                  color=base_color, alpha=0.6, edgecolor='black', linewidth=1.5)
            offset_val += width
            # ML Benchmark - darker shade
            if ml_colors:
                ax.bar(i + offset_val, ml_counts[i], width,
                      color=base_color, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Create legend with representative colors
        from matplotlib.patches import Patch
        legend_elements = []
        if gt_colors:
            legend_elements.append(Patch(facecolor='gray', alpha=1.0, edgecolor='black', hatch='//', label='Ground Truth'))
        legend_elements.extend([
            Patch(facecolor='gray', alpha=0.4, edgecolor='black', label='Quick Analysis'),
            Patch(facecolor='gray', alpha=0.6, edgecolor='black', label='Deep Analysis')
        ])
        if ml_colors:
            legend_elements.append(Patch(facecolor='gray', alpha=0.8, edgecolor='black', label='ML Benchmark'))
        
        ax.set_xlabel('Color', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Color Distribution: Quick vs Deep vs ML vs Ground Truth', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_colors, rotation=45, ha='right')
        ax.legend(handles=legend_elements, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        all_counts = gt_counts + quick_counts + deep_counts + ml_counts
        max_count = max(all_counts) if all_counts else 1
        offset = -1.5 * width
        for i, (gc, qc, dc, mc) in enumerate(zip(gt_counts, quick_counts, deep_counts, ml_counts)):
            if gc > 0 and gt_colors:
                ax.text(i + offset, gc + max_count * 0.02, 
                       str(gc), ha='center', va='bottom', fontsize=8, fontweight='bold')
            offset_tmp = offset + width
            if qc > 0:
                ax.text(i + offset_tmp, qc + max_count * 0.02, 
                       str(qc), ha='center', va='bottom', fontsize=8, fontweight='bold')
            offset_tmp += width
            if dc > 0:
                ax.text(i + offset_tmp, dc + max_count * 0.02, 
                       str(dc), ha='center', va='bottom', fontsize=8, fontweight='bold')
            offset_tmp += width
            if mc > 0 and ml_colors:
                ax.text(i + offset_tmp, mc + max_count * 0.02, 
                       str(mc), ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_area_comparison_chart(self, quick_areas: List, deep_areas: List, ml_areas: List = None, gt_areas: List = None) -> str:
        """Create area distribution comparison chart for Quick vs Deep vs ML vs Ground Truth"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if ml_areas is None:
            ml_areas = []
        if gt_areas is None:
            gt_areas = []
        
        if not quick_areas and not deep_areas and not ml_areas and not gt_areas:
            return ""
        
        all_areas = gt_areas + quick_areas + deep_areas + ml_areas
        if all_areas:
            bins = np.linspace(min(all_areas), max(all_areas), 20)
            
            if gt_areas:
                ax.hist(gt_areas, bins=bins, alpha=0.4, label='Ground Truth', 
                       color='#FF5722', edgecolor='black', histtype='step', linewidth=3)
            ax.hist(quick_areas, bins=bins, alpha=0.4, label='Quick Analysis', 
                   color='#2196F3', edgecolor='black')
            ax.hist(deep_areas, bins=bins, alpha=0.4, label='Deep Analysis', 
                   color='#4CAF50', edgecolor='black')
            if ml_areas:
                ax.hist(ml_areas, bins=bins, alpha=0.4, label='ML Benchmark', 
                       color='#9C27B0', edgecolor='black')
            
            # Add mean lines
            if gt_areas:
                gt_mean = np.mean(gt_areas)
                ax.axvline(gt_mean, color='#FF5722', linestyle='-.', linewidth=3, 
                          label=f'GT Mean: {gt_mean:.1f}')
            if quick_areas:
                quick_mean = np.mean(quick_areas)
                ax.axvline(quick_mean, color='#2196F3', linestyle='--', linewidth=2, 
                          label=f'Quick Mean: {quick_mean:.1f}')
            if deep_areas:
                deep_mean = np.mean(deep_areas)
                ax.axvline(deep_mean, color='#4CAF50', linestyle='--', linewidth=2, 
                          label=f'Deep Mean: {deep_mean:.1f}')
            if ml_areas:
                ml_mean = np.mean(ml_areas)
                ax.axvline(ml_mean, color='#9C27B0', linestyle='--', linewidth=2, 
                          label=f'ML Mean: {ml_mean:.1f}')
            
            ax.set_xlabel('Area (pixels²)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title('Object Pixel Area Distribution: Quick vs Deep vs ML vs Ground Truth', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_area_distribution_chart(self, area_data: Dict) -> str:
        """Create area distribution histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract areas (could be list or dict with bins)
        if isinstance(area_data, dict) and 'bins' in area_data:
            bins = area_data['bins']
            counts = area_data['counts']
            ax.bar(range(len(counts)), counts, color='#03A9F4', alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(bins)-1))
            ax.set_xticklabels([f'{bins[i]:.0f}-{bins[i+1]:.0f}' for i in range(len(bins)-1)], 
                              rotation=45, ha='right')
        elif isinstance(area_data, list):
            ax.hist(area_data, bins=30, color='#03A9F4', alpha=0.7, edgecolor='black')
            mean_area = np.mean(area_data)
            median_area = np.median(area_data)
            ax.axvline(mean_area, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_area:.1f}')
            ax.axvline(median_area, color='green', linestyle='--', linewidth=2, 
                      label=f'Median: {median_area:.1f}')
            ax.legend()
        
        ax.set_xlabel('Area (pixels²)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Particle Area Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_shape_distribution_chart(self, shape_data: Dict) -> str:
        """Create shape distribution pie chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        labels = list(shape_data.keys())
        sizes = list(shape_data.values())
        
        colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF']
        
        # Pie chart
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                            colors=colors, startangle=90)
        ax1.set_title('Shape Distribution (Pie Chart)', fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        # Bar chart
        ax2.bar(labels, sizes, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Shape Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('Shape Distribution (Bar Chart)', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, val in enumerate(sizes):
            ax2.text(i, val + max(sizes) * 0.02, str(val), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_color_distribution_chart(self, color_data: Dict) -> str:
        """Create color distribution chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = list(color_data.keys())
        sizes = list(color_data.values())
        
        # Map color names to actual colors
        color_map = {
            'Red': '#FF0000', 'Green': '#00FF00', 'Blue': '#0000FF',
            'Yellow': '#FFFF00', 'Orange': '#FFA500', 'Purple': '#800080',
            'Pink': '#FFC0CB', 'Brown': '#A52A2A', 'Black': '#000000',
            'White': '#FFFFFF', 'Gray': '#808080', 'Cyan': '#00FFFF',
            'Magenta': '#FF00FF', 'Lime': '#00FF00', 'Navy': '#000080'
        }
        
        bar_colors = [color_map.get(label, '#CCCCCC') for label in labels]
        
        bars = ax.bar(labels, sizes, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Color', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Color Distribution', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(sizes) * 0.02,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_confusion_matrix_chart(self, confusion_data: Dict) -> str:
        """Create confusion matrix heatmap"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Assuming confusion_data has 'matrix', 'labels' keys
        matrix = np.array(confusion_data.get('matrix', [[0]]))
        labels = confusion_data.get('labels', ['TP', 'FP', 'FN', 'TN'])
        
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, matrix[i, j], ha="center", va="center", 
                             color="white" if matrix[i, j] > matrix.max()/2 else "black",
                             fontsize=14, fontweight='bold')
        
        ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    def _build_html(self, results: Dict, charts: Dict) -> str:
        """Build complete HTML document"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .summary-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .summary-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        
        .summary-card h3 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 1.2em;
        }}
        
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }}
        
        .summary-card .label {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #333;
            border-left: 5px solid #4CAF50;
            padding-left: 15px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .chart-container {{
            background: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        th {{
            background: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        tr:hover {{
            background: #f5f5f5;
        }}
        
        .metric {{
            display: inline-block;
            background: #e3f2fd;
            padding: 5px 15px;
            border-radius: 20px;
            margin: 5px;
            font-weight: bold;
            color: #1976D2;
        }}
        
        .good {{ color: #4CAF50; }}
        .medium {{ color: #FF9800; }}
        .bad {{ color: #F44336; }}
        
        footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Microplastic Benchmark Analysis Report</h1>
            <p>Generated on {timestamp}</p>
        </header>
        
        <div class="content">
"""
        
        # Summary cards
        quick_data = results.get('quick_analysis', {})
        deep_data = results.get('deep_analysis', {})
        ml_data = results.get('ml_benchmark', {})
        num_images = results.get('num_images', 0)
        
        html += f"""
            <div class="summary">
                <div class="summary-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                    <h3 style="color: white;">Benchmark Summary</h3>
                    <div class="value" style="color: white;">{num_images}</div>
                    <div class="label" style="color: white;">Images Analyzed</div>
                </div>
"""
        
        # Add summary cards for each method
        for method_name, method_data in [('Quick Analysis', quick_data), 
                                         ('Deep Analysis', deep_data),
                                         ('ML Benchmark', ml_data)]:
            if method_data:
                precision = method_data.get('precision', 0)
                recall = method_data.get('recall', 0)
                f1 = method_data.get('f1_score', 0)
                detected = method_data.get('detected', 0)
                boundary_removed = method_data.get('boundary_objects_removed', 0)
                
                boundary_info = ""
                if boundary_removed > 0:
                    boundary_info = f'<div style="margin-top: 5px; font-size: 0.85em; color: #666;"><em>({boundary_removed} boundary objects excluded)</em></div>'
                
                html += f"""
                <div class="summary-card">
                    <h3>{method_name}</h3>
                    <div class="value">{detected}</div>
                    <div class="label">Avg Particles per Image</div>
                    {boundary_info}
                    <div style="margin-top: 10px;">
                        <span class="metric">Precision: {precision:.2f}</span>
                        <span class="metric">Recall: {recall:.2f}</span>
                        <span class="metric">F1: {f1:.2f}</span>
                    </div>
                </div>
"""
        
        # Ground truth card
        ground_truth = deep_data.get('ground_truth', 0)
        html += f"""
                <div class="summary-card">
                    <h3>Ground Truth</h3>
                    <div class="value">{ground_truth}</div>
                    <div class="label">Avg True Particles per Image</div>
                </div>
            </div>
"""
        
        # Add batch summary section
        if num_images > 1:
            total_quick = quick_data.get('detected', 0) * num_images
            total_deep = deep_data.get('detected', 0) * num_images
            total_ml = ml_data.get('detected', 0) * num_images if ml_data else 0
            total_gt = ground_truth * num_images
            
            ml_summary = f"""
                        <li><strong>ML Benchmark:</strong> ~{total_ml:.0f} particles total ({ml_data.get('detected', 0):.1f} avg per image)</li>
""" if ml_data else ""
            
            html += f"""
            <div class="section">
                <h2>📈 Batch Analysis Summary</h2>
                <div style="background: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                    <h3 style="margin-top: 0;">Total Particles Detected Across All {num_images} Images:</h3>
                    <ul style="font-size: 1.1em; line-height: 2;">
                        <li><strong>Quick Analysis:</strong> ~{total_quick:.0f} particles total ({quick_data.get('detected', 0):.1f} avg per image)</li>
                        <li><strong>Deep Analysis:</strong> ~{total_deep:.0f} particles total ({deep_data.get('detected', 0):.1f} avg per image)</li>
                        {ml_summary}<li><strong>Ground Truth:</strong> ~{total_gt:.0f} particles total ({ground_truth:.1f} avg per image)</li>
                    </ul>
                    <p style="margin-top: 15px; color: #666;">
                        <em>Note: The precision, recall, and F1 scores shown are calculated by aggregating 
                        true positives, false positives, and false negatives across all images.</em>
                    </p>
                </div>
            </div>
"""
        
        # Charts sections
        if 'precision_recall' in charts and charts['precision_recall']:
            html += f"""
            <div class="section">
                <h2>📊 Performance Metrics</h2>
                <div class="chart-container">
                    <img src="{charts['precision_recall']}" alt="Precision/Recall Chart">
                </div>
            </div>
"""
        
        # Comparison charts (Quick vs Deep vs ML vs Ground Truth)
        if 'shape_comparison' in charts and charts['shape_comparison']:
            html += f"""
            <div class="section">
                <h2>🔷 Shape Distribution Comparison (Quick vs Deep vs ML vs Ground Truth)</h2>
                <div class="chart-container">
                    <img src="{charts['shape_comparison']}" alt="Shape Distribution Comparison">
                </div>
            </div>
"""
        
        if 'color_comparison' in charts and charts['color_comparison']:
            html += f"""
            <div class="section">
                <h2>🎨 Color Distribution Comparison (Quick vs Deep vs ML vs Ground Truth)</h2>
                <div class="chart-container">
                    <img src="{charts['color_comparison']}" alt="Color Distribution Comparison">
                </div>
            </div>
"""
        
        if 'area_comparison' in charts and charts['area_comparison']:
            html += f"""
            <div class="section">
                <h2>📏 Object Pixel Area Distribution Comparison (Quick vs Deep vs ML vs Ground Truth)</h2>
                <div class="chart-container">
                    <img src="{charts['area_comparison']}" alt="Area Distribution Comparison">
                </div>
            </div>
"""
        
        # Legacy charts (if no comparison charts)
        if 'area_dist' in charts and charts['area_dist'] and 'area_comparison' not in charts:
            html += f"""
            <div class="section">
                <h2>📏 Area Distribution</h2>
                <div class="chart-container">
                    <img src="{charts['area_dist']}" alt="Area Distribution">
                </div>
            </div>
"""
        
        if 'shape_dist' in charts and charts['shape_dist'] and 'shape_comparison' not in charts:
            html += f"""
            <div class="section">
                <h2>🔷 Shape Distribution</h2>
                <div class="chart-container">
                    <img src="{charts['shape_dist']}" alt="Shape Distribution">
                </div>
            </div>
"""
        
        if 'color_dist' in charts and charts['color_dist'] and 'color_comparison' not in charts:
            html += f"""
            <div class="section">
                <h2>🎨 Color Distribution</h2>
                <div class="chart-container">
                    <img src="{charts['color_dist']}" alt="Color Distribution">
                </div>
            </div>
"""
        
        # Detailed metrics table
        html += f"""
            <div class="section">
                <h2>📋 Detailed Metrics (Averaged Across {num_images} Images)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Method</th>
                            <th>Avg Detected</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>Avg Time/Image</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for method_name, method_data in [('Quick Analysis', quick_data), 
                                         ('Deep Analysis', deep_data),
                                         ('Watershed', results.get('watershed_analysis', {})),
                                         ('ML Benchmark', ml_data)]:
            if method_data:
                detected = method_data.get('detected', 0)
                precision = method_data.get('precision', 0)
                recall = method_data.get('recall', 0)
                f1 = method_data.get('f1_score', 0)
                time_val = method_data.get('processing_time', 0)
                
                precision_class = 'good' if precision >= 0.8 else ('medium' if precision >= 0.5 else 'bad')
                recall_class = 'good' if recall >= 0.8 else ('medium' if recall >= 0.5 else 'bad')
                f1_class = 'good' if f1 >= 0.8 else ('medium' if f1 >= 0.5 else 'bad')
                
                html += f"""
                        <tr>
                            <td><strong>{method_name}</strong></td>
                            <td>{detected}</td>
                            <td class="{precision_class}">{precision:.3f}</td>
                            <td class="{recall_class}">{recall:.3f}</td>
                            <td class="{f1_class}">{f1:.3f}</td>
                            <td>{time_val:.2f}s</td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <footer>
            <p>Microplastic Analyzer Pro - Benchmark Report</p>
            <p>© 2026 - Generated automatically</p>
        </footer>
    </div>
</body>
</html>
"""
        
        return html
    
    def _load_template(self) -> str:
        """Load HTML template (placeholder for future expansion)"""
        return ""
