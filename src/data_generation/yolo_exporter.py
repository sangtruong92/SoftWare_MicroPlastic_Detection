"""
YOLO Dataset Exporter
Exports synthetic microplastic images and annotations in YOLOv8 format
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import yaml

from config.constants import SHAPE_CATEGORIES


class YOLODatasetExporter:
    """Export synthetic images and ground truth in YOLOv8 format"""
    
    # Map individual shape names to grouped categories
    SHAPE_TO_CATEGORY = {
        'Microbead': 'Microbead/Pellet',
        'Pellet': 'Microbead/Pellet',
        'Fiber': 'Fiber/Filament',
        'Filament': 'Fiber/Filament',
        'Fragment': 'Fragment',
        'Irregular': 'Irregular'
    }
    
    @staticmethod
    def get_class_id(shape_name: str) -> int:
        """
        Get YOLO class ID for a shape name.
        
        Args:
            shape_name: Individual shape name (e.g., 'Fiber', 'Microbead')
            
        Returns:
            Class ID (0-4) corresponding to grouped category
        """
        # Map to grouped category
        category = YOLODatasetExporter.SHAPE_TO_CATEGORY.get(shape_name, 'Irregular')
        
        # Return index in SHAPE_CATEGORIES
        try:
            return SHAPE_CATEGORIES.index(category)
        except ValueError:
            return 4  # Default to 'Irregular' if not found
    
    @staticmethod
    def ground_truth_to_yolo_bbox(particle: Dict, image_width: int, image_height: int) -> Tuple[int, float, float, float, float]:
        """
        Convert ground truth particle data to YOLO format bounding box.
        
        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        All coordinates normalized to [0, 1]
        
        Args:
            particle: Ground truth dict with 'shape', 'position', 'size', 'area'
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Tuple of (class_id, x_center_norm, y_center_norm, width_norm, height_norm)
        """
        # Get class ID
        shape_name = particle.get('shape', 'Irregular')
        class_id = YOLODatasetExporter.get_class_id(shape_name)
        
        # Get position (center)
        cx, cy = particle['position']
        
        # Estimate bounding box size from area
        # For circles: area = π*r², so r = sqrt(area/π)
        # For other shapes, use approximation
        area = particle.get('area', 100)
        
        # Estimate radius/half-size from area
        # For safety, add 20% margin to ensure full object is captured
        estimated_radius = np.sqrt(area / np.pi) * 1.2
        
        # For elongated shapes (fibers/filaments), use size hint
        size_hint = particle.get('size', estimated_radius)
        
        # Determine if shape is elongated
        if 'Fiber' in shape_name or 'Filament' in shape_name:
            # For fibers, width is small, height is length
            bbox_width = max(size_hint * 0.3, 8)  # Minimum 8 pixels wide
            bbox_height = size_hint * 1.2
        else:
            # For other shapes, use estimated radius
            bbox_width = estimated_radius * 2
            bbox_height = estimated_radius * 2
        
        # Normalize to [0, 1]
        x_center_norm = cx / image_width
        y_center_norm = cy / image_height
        width_norm = bbox_width / image_width
        height_norm = bbox_height / image_height
        
        # Clamp to valid range
        x_center_norm = np.clip(x_center_norm, 0.0, 1.0)
        y_center_norm = np.clip(y_center_norm, 0.0, 1.0)
        width_norm = np.clip(width_norm, 0.01, 1.0)
        height_norm = np.clip(height_norm, 0.01, 1.0)
        
        return class_id, x_center_norm, y_center_norm, width_norm, height_norm
    
    @staticmethod
    def save_yolo_annotation(annotation_path: Path, particles: List[Dict], image_width: int, image_height: int):
        """
        Save YOLO format annotation file.
        
        Args:
            annotation_path: Path to save .txt annotation file
            particles: List of ground truth particle dicts
            image_width: Image width in pixels
            image_height: Image height in pixels
        """
        with open(annotation_path, 'w') as f:
            for particle in particles:
                class_id, x_c, y_c, w, h = YOLODatasetExporter.ground_truth_to_yolo_bbox(
                    particle, image_width, image_height
                )
                # Write in YOLO format: class x_center y_center width height
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
    
    @staticmethod
    def save_detailed_ground_truth(ground_truth_path: Path, particles: List[Dict]):
        """
        Save detailed ground truth information (like benchmark format).
        
        Args:
            ground_truth_path: Path to save detailed ground truth .txt file
            particles: List of ground truth particle dicts
        """
        with open(ground_truth_path, 'w') as f:
            f.write(f"Total Particles: {len(particles)}\n\n")
            
            for i, particle in enumerate(particles, 1):
                f.write(f"Particle {i}:\n")
                f.write(f"  Shape: {particle.get('shape', 'Unknown')}\n")
                f.write(f"  Color: {particle.get('color', 'Unknown')}\n")
                
                if 'wavelength' in particle and particle['wavelength']:
                    f.write(f"  Wavelength: {particle['wavelength']}nm\n")
                
                position = particle.get('position', (0, 0))
                f.write(f"  Position: ({position[0]:.1f}, {position[1]:.1f})\n")
                f.write(f"  Area: {particle.get('area', 0):.1f} px²\n")
                f.write(f"  Size: {particle.get('size', 0):.1f} px\n")
                f.write("\n")
    
    @staticmethod
    def save_single_image(image: np.ndarray, ground_truth: List[Dict], name: str,
                         images_dir: Path, labels_dir: Path, ground_truth_dir: Path):
        """
        Save a single image with its YOLO annotation and ground truth immediately to disk.
        
        Args:
            image: Image array (RGB format)
            ground_truth: List of ground truth particle dicts
            name: Base name for the files (e.g., 'train_00001')
            images_dir: Directory to save image
            labels_dir: Directory to save YOLO label
            ground_truth_dir: Directory to save detailed ground truth
        """
        if not ground_truth:
            return
        
        # Save image
        image_filename = f"{name}.png"
        image_path = images_dir / image_filename
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        cv2.imwrite(str(image_path), image_bgr)
        
        # Save YOLO annotation
        annotation_filename = f"{name}.txt"
        annotation_path = labels_dir / annotation_filename
        
        image_height, image_width = image.shape[:2]
        YOLODatasetExporter.save_yolo_annotation(
            annotation_path, ground_truth, image_width, image_height
        )
        
        # Save detailed ground truth
        ground_truth_filename = f"{name}_groundtruth.txt"
        ground_truth_path = ground_truth_dir / ground_truth_filename
        YOLODatasetExporter.save_detailed_ground_truth(
            ground_truth_path, ground_truth
        )
    
    @staticmethod
    def export_dataset(images_data: List[Dict], output_dir: Path, split: str = 'train'):
        """
        Export full YOLO dataset with images and labels.
        
        Args:
            images_data: List of dicts with 'image', 'ground_truth_data', 'name'
            output_dir: Base output directory for dataset
            split: Dataset split ('train', 'val', or 'test')
        """
        # Create directory structure
        images_dir = output_dir / 'images' / split
        labels_dir = output_dir / 'labels' / split
        ground_truth_dir = output_dir / 'ground_truth' / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        ground_truth_dir.mkdir(parents=True, exist_ok=True)
        
        exported_count = 0
        
        for item in images_data:
            image = item['image']
            ground_truth = item.get('ground_truth_data', [])
            name = item.get('name', f'image_{exported_count:05d}')
            
            # Skip if no ground truth data
            if not ground_truth:
                continue
            
            # Save image
            image_filename = f"{name}.png"
            image_path = images_dir / image_filename
            
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            cv2.imwrite(str(image_path), image_bgr)
            
            # Save YOLO annotation (for training)
            annotation_filename = f"{name}.txt"
            annotation_path = labels_dir / annotation_filename
            
            image_height, image_width = image.shape[:2]
            YOLODatasetExporter.save_yolo_annotation(
                annotation_path, ground_truth, image_width, image_height
            )
            
            # Save detailed ground truth (like benchmark)
            ground_truth_filename = f"{name}_groundtruth.txt"
            ground_truth_path = ground_truth_dir / ground_truth_filename
            YOLODatasetExporter.save_detailed_ground_truth(
                ground_truth_path, ground_truth
            )
            
            exported_count += 1
        
        return exported_count
    
    @staticmethod
    def create_dataset_yaml(output_dir: Path, dataset_name: str = 'microplastic'):
        """
        Create dataset.yaml file for YOLOv8 training.
        
        Args:
            output_dir: Base dataset directory
            dataset_name: Name of the dataset
        """
        # Get absolute paths
        train_images = str((output_dir / 'images' / 'train').absolute())
        val_images = str((output_dir / 'images' / 'val').absolute())
        test_images = str((output_dir / 'images' / 'test').absolute())
        
        # Create dataset configuration
        dataset_config = {
            'path': str(output_dir.absolute()),  # Dataset root dir
            'train': train_images,  # Training images
            'val': val_images,      # Validation images
            'test': test_images,    # Test images (optional)
            'nc': len(SHAPE_CATEGORIES),  # Number of classes
            'names': SHAPE_CATEGORIES  # Class names
        }
        
        # Save as YAML
        yaml_path = output_dir / f'{dataset_name}.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
        
        return yaml_path
    
    @staticmethod
    def get_dataset_stats(output_dir: Path) -> Dict:
        """
        Get statistics about exported dataset.
        
        Args:
            output_dir: Base dataset directory
            
        Returns:
            Dict with counts per class and split
        """
        stats = {
            'train': {'total': 0, 'classes': {cat: 0 for cat in SHAPE_CATEGORIES}},
            'val': {'total': 0, 'classes': {cat: 0 for cat in SHAPE_CATEGORIES}},
            'test': {'total': 0, 'classes': {cat: 0 for cat in SHAPE_CATEGORIES}}
        }
        
        for split in ['train', 'val', 'test']:
            labels_dir = output_dir / 'labels' / split
            if not labels_dir.exists():
                continue
            
            for label_file in labels_dir.glob('*.txt'):
                stats[split]['total'] += 1
                
                # Count classes in this file
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if 0 <= class_id < len(SHAPE_CATEGORIES):
                                class_name = SHAPE_CATEGORIES[class_id]
                                stats[split]['classes'][class_name] += 1
        
        return stats


def print_dataset_stats(stats: Dict):
    """Print formatted dataset statistics."""
    print("\n" + "="*60)
    print("YOLO Dataset Statistics")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        if stats[split]['total'] > 0:
            print(f"\n{split.upper()} Split:")
            print(f"  Total images: {stats[split]['total']}")
            print(f"  Objects by class:")
            for class_name, count in stats[split]['classes'].items():
                if count > 0:
                    print(f"    {class_name}: {count}")
    
    print("\n" + "="*60)
