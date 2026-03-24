"""
Statistics Comparison Module
Compares statistical parameters across different analysis methods:
- Quick Analysis
- Deep Analysis  
- Synthetic Image Generation (Ground Truth)
- Machine Learning Detection (BenchMark_ML)
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from config.constants import SHAPE_GROUP_MAPPING


def map_to_grouped_shape(shape_name: str) -> str:
    """Map individual shape names to grouped categories"""
    return SHAPE_GROUP_MAPPING.get(shape_name, shape_name)


@dataclass
class StatisticsProfile:
    """Statistical profile for a single analysis method"""
    method_name: str  # 'quick', 'deep', 'synthetic_gt', 'ml_benchmark'
    
    # Core statistics
    circularity_values: List[float] = field(default_factory=list)
    aspect_ratio_values: List[float] = field(default_factory=list)
    eccentricity_values: List[float] = field(default_factory=list)
    area_values: List[float] = field(default_factory=list)
    
    # Advanced statistics (may not be available in all methods)
    rectangularity_values: List[float] = field(default_factory=list)
    solidity_values: List[float] = field(default_factory=list)
    perimeter_values: List[float] = field(default_factory=list)
    
    # Shape and color distributions
    shape_distribution: Dict[str, int] = field(default_factory=dict)
    color_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Summary statistics
    num_detections: int = 0
    processing_time: float = 0.0
    
    def compute_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute mean, std, min, max for each parameter"""
        summary = {}
        
        for param_name, values in [
            ('circularity', self.circularity_values),
            ('aspect_ratio', self.aspect_ratio_values),
            ('eccentricity', self.eccentricity_values),
            ('area', self.area_values),
            ('rectangularity', self.rectangularity_values),
            ('solidity', self.solidity_values),
            ('perimeter', self.perimeter_values)
        ]:
            if values:
                summary[param_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
            else:
                summary[param_name] = {
                    'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0
                }
        
        return summary


class StatisticsComparator:
    """Compare statistics across different analysis methods"""
    
    def __init__(self):
        self.profiles: Dict[str, StatisticsProfile] = {}
        
    def add_quick_analysis(self, result) -> StatisticsProfile:
        """Add quick analysis results"""
        profile = StatisticsProfile(method_name='Quick Analysis')
        profile.num_detections = result.num_detections
        profile.processing_time = result.processing_time
        
        for feature in result.features:
            profile.circularity_values.append(feature.get('circularity', 0))
            profile.aspect_ratio_values.append(feature.get('aspect_ratio', 1.0))
            profile.eccentricity_values.append(feature.get('eccentricity', 0))
            profile.area_values.append(feature.get('area', 0))
            profile.perimeter_values.append(feature.get('perimeter', 0))
            
            # Shape and color distribution
            shape = feature.get('shape', 'Unknown')
            color = feature.get('color', 'Unknown')
            profile.shape_distribution[shape] = profile.shape_distribution.get(shape, 0) + 1
            profile.color_distribution[color] = profile.color_distribution.get(color, 0) + 1
        
        self.profiles['quick'] = profile
        return profile
    
    def add_deep_analysis(self, result) -> StatisticsProfile:
        """Add deep analysis results"""
        profile = StatisticsProfile(method_name='Deep Analysis')
        profile.num_detections = result.num_detections
        profile.processing_time = result.processing_time
        
        for feature in result.features:
            profile.circularity_values.append(feature.get('circularity', 0))
            profile.aspect_ratio_values.append(feature.get('aspect_ratio', 1.0))
            profile.eccentricity_values.append(feature.get('eccentricity', 0))
            profile.area_values.append(feature.get('area', 0))
            profile.perimeter_values.append(feature.get('perimeter', 0))
            profile.rectangularity_values.append(feature.get('rectangularity', 0))
            profile.solidity_values.append(feature.get('solidity', 0))
            
            # Shape and color distribution
            shape = feature.get('shape', 'Unknown')
            color = feature.get('color', 'Unknown')
            profile.shape_distribution[shape] = profile.shape_distribution.get(shape, 0) + 1
            profile.color_distribution[color] = profile.color_distribution.get(color, 0) + 1
        
        self.profiles['deep'] = profile
        return profile
    
    def add_synthetic_ground_truth(self, ground_truth: List[Dict], processing_time: float = 0) -> StatisticsProfile:
        """Add synthetic image ground truth"""
        profile = StatisticsProfile(method_name='Synthetic Ground Truth')
        profile.num_detections = len(ground_truth)
        profile.processing_time = processing_time
        
        for particle in ground_truth:
            profile.circularity_values.append(particle.get('circularity', 0))
            profile.area_values.append(particle.get('area', 0))
            
            # Shape and color distribution - map to grouped names
            shape = map_to_grouped_shape(particle.get('shape', 'Unknown'))
            color_label = particle.get('color_label', 'Unknown')
            profile.shape_distribution[shape] = profile.shape_distribution.get(shape, 0) + 1
            profile.color_distribution[color_label] = profile.color_distribution.get(color_label, 0) + 1
        
        self.profiles['synthetic_gt'] = profile
        return profile
    
    def add_ml_benchmark(self, ml_result) -> StatisticsProfile:
        """Add ML benchmark results"""
        profile = StatisticsProfile(method_name='ML Benchmark')
        profile.num_detections = ml_result.num_detections
        profile.processing_time = ml_result.processing_time
        
        for feature in ml_result.features:
            profile.circularity_values.append(feature.get('circularity', 0))
            profile.aspect_ratio_values.append(feature.get('aspect_ratio', 1.0))
            profile.eccentricity_values.append(feature.get('eccentricity', 0))
            profile.area_values.append(feature.get('area', 0))
            profile.perimeter_values.append(feature.get('perimeter', 0))
            profile.rectangularity_values.append(feature.get('rectangularity', 0))
            profile.solidity_values.append(feature.get('solidity', 0))
            
            # Shape and color distribution
            shape = feature.get('shape', 'Unknown')
            color = feature.get('color', 'Unknown')
            profile.shape_distribution[shape] = profile.shape_distribution.get(shape, 0) + 1
            profile.color_distribution[color] = profile.color_distribution.get(color, 0) + 1
        
        self.profiles['ml_benchmark'] = profile
        return profile
    
    def get_comparison_table(self) -> Dict[str, Dict[str, any]]:
        """Generate comparison table of all methods"""
        comparison = {}
        
        for method_key, profile in self.profiles.items():
            summary = profile.compute_summary_stats()
            comparison[profile.method_name] = {
                'num_detections': profile.num_detections,
                'processing_time': f"{profile.processing_time:.3f}s",
                'statistics': summary,
                'shapes': profile.shape_distribution,
                'colors': profile.color_distribution
            }
        
        return comparison
    
    def get_parameter_comparison(self, parameter: str) -> Dict[str, List[float]]:
        """Get values of a specific parameter across all methods"""
        comparison = {}
        
        for method_key, profile in self.profiles.items():
            if parameter == 'circularity':
                comparison[profile.method_name] = profile.circularity_values
            elif parameter == 'aspect_ratio':
                comparison[profile.method_name] = profile.aspect_ratio_values
            elif parameter == 'eccentricity':
                comparison[profile.method_name] = profile.eccentricity_values
            elif parameter == 'area':
                comparison[profile.method_name] = profile.area_values
            elif parameter == 'rectangularity':
                comparison[profile.method_name] = profile.rectangularity_values
            elif parameter == 'solidity':
                comparison[profile.method_name] = profile.solidity_values
            elif parameter == 'perimeter':
                comparison[profile.method_name] = profile.perimeter_values
        
        return comparison
    
    def get_all_methods(self) -> List[str]:
        """Get list of all analysis methods in comparator"""
        return [profile.method_name for profile in self.profiles.values()]
    
    def clear(self):
        """Clear all stored profiles"""
        self.profiles.clear()
