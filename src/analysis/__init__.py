"""
Analysis Module
"""

from .quick_analyzer import QuickAnalyzer
from .deep_analyzer import DeepAnalyzer
from .report_generator import ReportGenerator
from .ml_benchmark_analyzer import MLBenchmarkAnalyzer
from .statistics_comparator import StatisticsComparator, StatisticsProfile

__all__ = ['QuickAnalyzer', 'DeepAnalyzer', 'ReportGenerator', 
           'MLBenchmarkAnalyzer', 'StatisticsComparator', 'StatisticsProfile']
