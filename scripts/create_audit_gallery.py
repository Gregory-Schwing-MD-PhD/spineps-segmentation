#!/usr/bin/env python3
"""
HTML Audit Gallery Generator
============================

Creates an interactive HTML page showing all correction visualizations
with sortable/filterable table of corrections.
"""

import argparse
import json
import base64
from pathlib import Path
from datetime import datetime

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LSTV Anatomical Correction Audit</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .stat-card .value { font-size: 2em; font-weight: 700; color: #2d3748; }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 30px;
            padding: 40px;
        }
        .case-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s ease;
        }
        .case-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }
        .case-header {
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 2px solid #667eea;
        }
        .case-header h3 { color: #2d3748; margin-bottom: 5px; }
        .case-meta { color: #718096; font-size: 0.9em; }
        .case-card img { width: 100%; display: block; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 40px;
            max-width: calc(100% - 80px);
        }
        thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        th, td { padding: 12px; text-align: left; }
        tbody tr { border-bottom: 1px solid #e2e8f0; }
        tbody tr:hover { background: #f7fafc; }
        .footer {
            background: #2d3748;
            color: white;
            padding: 30px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 LSTV Anatomical Correction Audit</h1>
            <p>Landmark-Based Vertebral Label Propagation</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Total Cases</h3>
                <div class="value">{{ stats.total_cases }}</div>
            </div>
            <div class="stat-card">
                <h3>Total Corrections</h3>
                <div class="value">{{ stats.total_corrections }}</div>
            </div>
            <div class="stat-card">
                <h3>Avg Corrections/Case</h3>
                <div class="value">{{ stats.avg_corrections }}</div>
            </div>
            <div class="stat-card">
                <h3>Max Corrections</h3>
                <div class="value">{{ stats.max_corrections }}</div>
            </div>
        </div>
        
        <div style="padding: 40px;">
            <h2 style="color: #2d3748; margin-bottom: 20px; font-size: 1.8em;">Case Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Study ID</th>
                        <th>Corrections</th>
                        <th>L5-S1 Entropy</th>
                        <th>Labels Changed</th>
                    </tr>
                </thead>
                <tbody>
                    {% for case in cases %}
                    <tr>
                        <td><strong>{{ loop.index }}</strong></td>
                        <td><a href="#case_{{ case.study_id }}">{{ case.study_id }}</a></td>
                        <td>{{ case.n_corrections }}</td>
                        <td>{{ "%.3f"|format(case.l5_s1_entropy) }}</td>
                        <td>{{ case.corrections|join(", ") }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div style="padding: 40px;">
            <h2 style="color: #2d3748; margin-bottom: 20px; font-size: 1.8em;">Visual Comparisons</h2>
            <div class="gallery">
                {% for case in cases %}
                <div class="case-card" id="case_{{ case.study_id }}">
                    <div class="case-header">
                        <h3>Study {{ case.study_id }}</h3>
                        <div class="case-meta">
                            {{ case.n_corrections }} corrections | 
                            L5-S1 H={{ "%.3f"|format(case.l5_s1_entropy) }}
                        </div>
                    </div>
                    <img src="data:image/png;base64,{{ case.image_b64 }}" alt="Case {{ case.study_id }}">
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="footer">
            <p>Generated: {{ timestamp }}</p>
            <p style="margin-top: 10px; font-size: 0.9em; opacity: 0.8;">
                Anatomical Label Propagation Pipeline | Wayne State University
            </p>
        </div>
    </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz_dir', required=True, help='Visualization images directory')
    parser.add_argument('--audit_json', required=True, help='Audit queue JSON')
    parser.add_argument('--output', required=True, help='Output HTML path')
    args = parser.parse_args()

    viz_dir = Path(args.viz_dir)
    
    # Load audit data
    with open(args.audit_json) as f:
        audit_data = json.load(f)
    
    # Embed images as base64
    cases = []
    total_corrections = 0
    max_corrections = 0
    
    for case in audit_data['cases']:
        study_id = case['study_id']
        img_path = viz_dir / f"{study_id}_comparison.png"
        
        if not img_path.exists():
            print(f"Warning: {img_path} not found")
            continue
        
        with open(img_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        cases.append({
            'study_id': study_id,
            'n_corrections': case['n_corrections'],
            'l5_s1_entropy': case.get('l5_s1_entropy', 0.0),
            'corrections': case.get('corrections', []),
            'image_b64': img_b64,
        })
        
        total_corrections += case['n_corrections']
        max_corrections = max(max_corrections, case['n_corrections'])
    
    stats = {
        'total_cases': len(cases),
        'total_corrections': total_corrections,
        'avg_corrections': f"{total_corrections / len(cases):.1f}" if cases else "0",
        'max_corrections': max_corrections,
    }
    
    # Render HTML
    from jinja2 import Template
    template = Template(HTML_TEMPLATE)
    html = template.render(
        stats=stats,
        cases=cases,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    Path(args.output).write_text(html)
    print(f"✓ Audit gallery saved to: {args.output}")
    print(f"✓ {len(cases)} cases with visualizations")
    print(f"\nOpen in browser:")
    print(f"  firefox {args.output}")


if __name__ == '__main__':
    main()
