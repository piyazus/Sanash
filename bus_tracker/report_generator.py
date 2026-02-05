"""
Report Generator for Bus Person Detection System
=================================================

Generates comprehensive reports in multiple formats:
- HTML report with embedded charts and summary
- CSV export for spreadsheet analysis
- JSON export for programmatic access

Usage:
    from bus_tracker.report_generator import ReportGenerator
    
    generator = ReportGenerator(output_dir="output/reports")
    generator.generate_html_report(analytics_data, video_info)
    generator.generate_csv_report(analytics_data)
    generator.generate_json_export(analytics_data)
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from . import config


# HTML Report Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bus Tracker Report - {{ report_date }}</title>
    <style>
        :root {
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-700: #374151;
            --gray-900: #111827;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--gray-100);
            color: var(--gray-900);
            line-height: 1.5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            background: linear-gradient(135deg, var(--primary), #1e40af);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        
        header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        header p {
            opacity: 0.9;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
        }
        
        .stat-label {
            color: var(--gray-700);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .card h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: var(--gray-900);
            border-bottom: 2px solid var(--gray-200);
            padding-bottom: 0.5rem;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
        }
        
        th {
            background: var(--gray-50);
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .chart-container {
            height: 300px;
            position: relative;
        }
        
        .bar-chart {
            display: flex;
            align-items: flex-end;
            justify-content: space-around;
            height: 200px;
            padding: 1rem;
            background: var(--gray-50);
            border-radius: 8px;
        }
        
        .bar {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 60px;
        }
        
        .bar-fill {
            width: 40px;
            background: linear-gradient(to top, var(--primary), #3b82f6);
            border-radius: 4px 4px 0 0;
            min-height: 4px;
        }
        
        .bar-label {
            margin-top: 0.5rem;
            font-size: 0.75rem;
            color: var(--gray-700);
            text-align: center;
        }
        
        .bar-value {
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        
        .heatmap-container {
            text-align: center;
        }
        
        .heatmap-container img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--gray-700);
            font-size: 0.875rem;
        }
        
        @media print {
            body { background: white; }
            .container { max-width: none; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚌 Bus Tracker Detection Report</h1>
            <p>Generated: {{ report_date }} | Video: {{ video_name }}</p>
        </header>
        
        <!-- Summary Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ summary.total_unique_people }}</div>
                <div class="stat-label">Unique People</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ summary.peak_occupancy }}</div>
                <div class="stat-label">Peak Occupancy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ "%.1f"|format(summary.average_dwell_time_seconds) }}s</div>
                <div class="stat-label">Avg Dwell Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ summary.entries }}</div>
                <div class="stat-label">Entries</div>
            </div>
        </div>
        
        <!-- Video Info -->
        <div class="card">
            <h2>📹 Video Information</h2>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Duration</td><td>{{ video_info.duration_formatted }}</td></tr>
                <tr><td>Resolution</td><td>{{ video_info.width }}x{{ video_info.height }}</td></tr>
                <tr><td>FPS</td><td>{{ "%.1f"|format(video_info.fps) }}</td></tr>
                <tr><td>Total Frames</td><td>{{ video_info.total_frames }}</td></tr>
                <tr><td>Peak Time</td><td>{{ summary.peak_occupancy_time }}</td></tr>
            </table>
        </div>
        
        <!-- Dwell Time Distribution -->
        <div class="card">
            <h2>⏱️ Dwell Time Distribution</h2>
            <div class="bar-chart">
                {% for label, count in dwell_distribution.items() %}
                <div class="bar">
                    <div class="bar-value">{{ count }}</div>
                    <div class="bar-fill" style="height: {{ (count / max_dwell * 150)|int }}px;"></div>
                    <div class="bar-label">{{ label }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Occupancy Over Time -->
        <div class="card">
            <h2>📊 Occupancy Over Time</h2>
            <div class="bar-chart">
                {% for slot in occupancy_time[:15] %}
                <div class="bar">
                    <div class="bar-value">{{ "%.0f"|format(slot.avg_count) }}</div>
                    <div class="bar-fill" style="height: {{ (slot.avg_count / max_occupancy * 150)|int }}px;"></div>
                    <div class="bar-label">{{ slot.time }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Heatmap -->
        {% if heatmap_path %}
        <div class="card">
            <h2>🔥 Congregation Heatmap</h2>
            <div class="heatmap-container">
                <img src="{{ heatmap_path }}" alt="Heatmap showing where people congregate most">
                <p style="margin-top: 1rem; color: var(--gray-700);">
                    Warmer colors indicate areas where people spend more time.
                </p>
            </div>
        </div>
        {% endif %}
        
        <footer>
            <p>Generated by Bus Tracker Person Detection System</p>
        </footer>
    </div>
</body>
</html>
"""


class ReportGenerator:
    """
    Generate reports from analytics data.
    
    Supports HTML, CSV, and JSON output formats.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir) if output_dir else config.OUTPUT_REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_html_report(
        self,
        analytics_data: Dict[str, Any],
        video_info: Dict[str, Any],
        heatmap_image: np.ndarray = None,
        output_name: str = None
    ) -> str:
        """
        Generate HTML report with charts and summary.
        
        Args:
            analytics_data: Analytics report from AnalyticsEngine
            video_info: Video metadata dictionary
            heatmap_image: Optional heatmap image (BGR numpy array)
            output_name: Optional custom output filename
            
        Returns:
            Path to generated HTML file
        """
        if not JINJA2_AVAILABLE:
            raise ImportError("jinja2 is required for HTML reports. Install with: pip install jinja2")
        
        # Save heatmap if provided
        heatmap_path = None
        if heatmap_image is not None:
            import cv2
            heatmap_filename = "heatmap.png"
            heatmap_full_path = self.output_dir / heatmap_filename
            cv2.imwrite(str(heatmap_full_path), heatmap_image)
            heatmap_path = heatmap_filename
        
        # Prepare template data
        summary = analytics_data.get('summary', {})
        dwell_dist = analytics_data.get('dwell_time_distribution', {})
        occupancy_time = analytics_data.get('occupancy_over_time', [])
        
        # Calculate max values for chart scaling
        max_dwell = max(dwell_dist.values()) if dwell_dist else 1
        max_occupancy = max([s.get('avg_count', 0) for s in occupancy_time]) if occupancy_time else 1
        
        template_data = {
            'report_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'video_name': video_info.get('filename', 'Unknown'),
            'video_info': video_info,
            'summary': summary,
            'dwell_distribution': dwell_dist,
            'max_dwell': max(max_dwell, 1),
            'occupancy_time': occupancy_time,
            'max_occupancy': max(max_occupancy, 1),
            'heatmap_path': heatmap_path
        }
        
        # Render template
        template = Template(HTML_TEMPLATE)
        html_content = template.render(**template_data)
        
        # Write file
        output_name = output_name or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path = self.output_dir / output_name
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def generate_csv_report(
        self,
        analytics_data: Dict[str, Any],
        output_name: str = None
    ) -> str:
        """
        Generate CSV report for spreadsheet analysis.
        
        Args:
            analytics_data: Analytics report from AnalyticsEngine
            output_name: Optional custom output filename
            
        Returns:
            Path to generated CSV file
        """
        output_name = output_name or f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = self.output_dir / output_name
        
        # Flatten data for CSV
        rows = []
        
        # Summary row
        summary = analytics_data.get('summary', {})
        rows.append({
            'type': 'summary',
            'metric': 'total_unique_people',
            'value': summary.get('total_unique_people', 0)
        })
        rows.append({
            'type': 'summary',
            'metric': 'peak_occupancy',
            'value': summary.get('peak_occupancy', 0)
        })
        rows.append({
            'type': 'summary',
            'metric': 'peak_time',
            'value': summary.get('peak_occupancy_time', '')
        })
        rows.append({
            'type': 'summary',
            'metric': 'avg_dwell_time_seconds',
            'value': summary.get('average_dwell_time_seconds', 0)
        })
        
        # Dwell distribution
        for bucket, count in analytics_data.get('dwell_time_distribution', {}).items():
            rows.append({
                'type': 'dwell_distribution',
                'metric': bucket,
                'value': count
            })
        
        # Write CSV
        if rows:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['type', 'metric', 'value'])
                writer.writeheader()
                writer.writerows(rows)
        
        return str(output_path)
    
    def generate_detailed_csv(
        self,
        analytics_data: Dict[str, Any],
        output_name: str = None
    ) -> str:
        """
        Generate detailed CSV with per-person tracking data.
        
        Args:
            analytics_data: Analytics report from AnalyticsEngine
            output_name: Optional custom output filename
            
        Returns:
            Path to generated CSV file
        """
        output_name = output_name or f"tracking_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = self.output_dir / output_name
        
        person_details = analytics_data.get('person_details', {})
        
        rows = []
        for track_id, details in person_details.items():
            rows.append({
                'track_id': track_id,
                'first_seen': details.get('first_seen', ''),
                'last_seen': details.get('last_seen', ''),
                'dwell_time_seconds': details.get('dwell_time_seconds', 0),
                'entered_via': details.get('entered_via', ''),
                'exited_via': details.get('exited_via', '')
            })
        
        if rows:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f, 
                    fieldnames=['track_id', 'first_seen', 'last_seen', 
                               'dwell_time_seconds', 'entered_via', 'exited_via']
                )
                writer.writeheader()
                writer.writerows(rows)
        
        return str(output_path)
    
    def generate_json_export(
        self,
        analytics_data: Dict[str, Any],
        video_info: Dict[str, Any] = None,
        output_name: str = None
    ) -> str:
        """
        Export full analytics data as JSON.
        
        Args:
            analytics_data: Analytics report from AnalyticsEngine
            video_info: Optional video metadata
            output_name: Optional custom output filename
            
        Returns:
            Path to generated JSON file
        """
        output_name = output_name or f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = self.output_dir / output_name
        
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator': 'Bus Tracker Person Detection System'
            },
            'video_info': video_info or {},
            'analytics': analytics_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return str(output_path)
    
    def generate_all_reports(
        self,
        analytics_data: Dict[str, Any],
        video_info: Dict[str, Any],
        heatmap_image: np.ndarray = None,
        base_name: str = None
    ) -> Dict[str, str]:
        """
        Generate all report types at once.
        
        Args:
            analytics_data: Analytics report from AnalyticsEngine
            video_info: Video metadata
            heatmap_image: Optional heatmap image
            base_name: Base name for output files
            
        Returns:
            Dictionary mapping report type to file path
        """
        base = base_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        paths = {}
        
        if config.GENERATE_HTML_REPORT:
            paths['html'] = self.generate_html_report(
                analytics_data, video_info, heatmap_image,
                f"report_{base}.html"
            )
        
        if config.GENERATE_CSV_REPORT:
            paths['csv_summary'] = self.generate_csv_report(
                analytics_data,
                f"summary_{base}.csv"
            )
            paths['csv_details'] = self.generate_detailed_csv(
                analytics_data,
                f"details_{base}.csv"
            )
        
        if config.GENERATE_JSON_EXPORT:
            paths['json'] = self.generate_json_export(
                analytics_data, video_info,
                f"export_{base}.json"
            )
        
        return paths
