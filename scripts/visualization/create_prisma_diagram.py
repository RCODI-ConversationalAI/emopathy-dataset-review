#!/usr/bin/env python3
"""
PRISMA 2020 Flow Diagram Generator - Updated Jan 15, 2026
Includes Empathy branch processing
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(18, 26))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Colors
color_id = '#DEEBF7'       # Blue - Identification
color_screen = '#FFF2CC'   # Yellow - Screening
color_elig = '#D5E8D4'     # Green - Eligibility
color_incl = '#E1D5E7'     # Purple - Included
color_excl = '#F8CECC'     # Red - Excluded
border_color = '#333333'

def draw_box(ax, x, y, w, h, text, color, fontsize=10):
    """Draw a rounded box with centered text"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.01,rounding_size=0.3",
                         facecolor=color, edgecolor=border_color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center', fontsize=fontsize,
            linespacing=1.4)

def draw_arrow_down(ax, x, y1, y2):
    """Draw vertical arrow"""
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

def draw_arrow_right(ax, x1, x2, y):
    """Draw horizontal arrow"""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

def draw_arrow_diag(ax, x1, y1, x2, y2):
    """Draw diagonal arrow"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Title
ax.text(50, 98, 'PRISMA 2020 Flow Diagram', ha='center', fontsize=18, fontweight='bold')
ax.text(50, 96, 'Emotion/Empathy Dataset Systematic Review', ha='center', fontsize=14)

# ============ IDENTIFICATION ============
ax.text(3, 93, 'Identification', fontsize=13, fontweight='bold', color='#1565C0')

# Records identified
draw_box(ax, 8, 85, 35, 7,
         'Records identified from databases\n(n = 4,133)\n\nACL: 1,040 | Scopus: 1,303 | ACM: 60\nWoS: 34 | Google Scholar: 214',
         color_id, fontsize=9)

# Duplicates removed
draw_box(ax, 55, 85, 28, 7,
         'Duplicate records removed\n(n = 1,482)',
         color_excl, fontsize=9)

# Arrow: identified -> duplicates
draw_arrow_right(ax, 43, 55, 88.5)

# ============ SCREENING ============
ax.text(3, 81, 'Screening', fontsize=13, fontweight='bold', color='#E65100')

# Arrow down
draw_arrow_down(ax, 25.5, 85, 79)

# Records screened
draw_box(ax, 8, 72, 35, 7,
         'Records screened\n(title/abstract)\n(n = 2,651)',
         color_screen, fontsize=9)

# Records excluded
draw_box(ax, 55, 72, 28, 7,
         'Records excluded\n(n = 345)\n\nEC1-EC4, EC6 criteria',
         color_excl, fontsize=9)

# Arrow: screened -> excluded
draw_arrow_right(ax, 43, 55, 75.5)

# ============ ELIGIBILITY ============
ax.text(3, 68, 'Eligibility', fontsize=13, fontweight='bold', color='#2E7D32')

# Arrow down
draw_arrow_down(ax, 25.5, 72, 66)

# Records for eligibility
draw_box(ax, 8, 59, 35, 7,
         'Records assessed for eligibility\n(n = 2,306)',
         color_elig, fontsize=9)

# Arrow down and split
draw_arrow_down(ax, 25.5, 59, 56)

# Horizontal line for split
ax.plot([25.5, 25.5], [56, 55], 'k-', lw=1.5)
ax.plot([20, 70], [55, 55], 'k-', lw=1.5)
ax.plot([20, 20], [55, 53], 'k-', lw=1.5)
ax.plot([70, 70], [55, 53], 'k-', lw=1.5)

# Arrow tips for split
draw_arrow_down(ax, 20, 53.5, 52)
draw_arrow_down(ax, 70, 53.5, 52)

# EMOTION branch
draw_box(ax, 5, 44, 30, 8,
         'EMOTION\n(n = 2,174)',
         color_elig, fontsize=10)

# EMPATHY branch
draw_box(ax, 55, 44, 30, 8,
         'EMPATHY\n(n = 132)',
         color_elig, fontsize=10)

# Exclusion arrows and boxes
draw_arrow_right(ax, 35, 42, 48)
draw_box(ax, 42, 45.5, 11, 5,
         'BERTopic\nexcluded\n(n = 301)',
         color_excl, fontsize=8)

draw_arrow_right(ax, 85, 92, 48)
draw_box(ax, 92, 44, 6, 8,
         'Excluded\n(n = 20)\n\nmanual\nscreening',
         color_excl, fontsize=6)

# ============ INCLUDED ============
ax.text(3, 41, 'Included', fontsize=13, fontweight='bold', color='#6A1B9A')

# Arrows down from branches
draw_arrow_down(ax, 20, 44, 40)
draw_arrow_down(ax, 70, 44, 40)

# Processing steps for EMOTION
draw_box(ax, 5, 32, 30, 8,
         'After processing\n\nBERTopic: -301\n2025 ACL: +208\nNon-English: -245',
         '#C8E6C9', fontsize=9)

# Processing steps for EMPATHY
draw_box(ax, 55, 32, 30, 8,
         'After processing\n\nManual screening: -20\n2025 ACL: +38\nNon-English: -3',
         '#C8E6C9', fontsize=9)

# Arrows down
draw_arrow_down(ax, 20, 32, 29)
draw_arrow_down(ax, 70, 32, 29)

# Final Emotion
draw_box(ax, 5, 21, 30, 8,
         'English Emotion Papers\n(n = 1,786)',
         color_incl, fontsize=10)

# Final Empathy
draw_box(ax, 55, 21, 30, 8,
         'English Empathy Papers\n(n = 147)',
         color_incl, fontsize=10)

# Arrow down from Emotion to classification
draw_arrow_down(ax, 20, 21, 18)

# Horizontal split for Emotion classification
ax.plot([20, 20], [18, 17], 'k-', lw=1.5)
ax.plot([10, 48], [17, 17], 'k-', lw=1.5)
ax.plot([10, 10], [17, 15], 'k-', lw=1.5)
ax.plot([29, 29], [17, 15], 'k-', lw=1.5)
ax.plot([48, 48], [17, 15], 'k-', lw=1.5)

draw_arrow_down(ax, 10, 15.5, 14)
draw_arrow_down(ax, 29, 15.5, 14)
draw_arrow_down(ax, 48, 15.5, 14)

# Arrow down from Empathy to classification
draw_arrow_down(ax, 70, 21, 18)

# Horizontal split for Empathy classification
ax.plot([70, 70], [18, 17], 'k-', lw=1.5)
ax.plot([62, 78], [17, 17], 'k-', lw=1.5)
ax.plot([62, 62], [17, 15], 'k-', lw=1.5)
ax.plot([78, 78], [17, 15], 'k-', lw=1.5)

draw_arrow_down(ax, 62, 15.5, 14)
draw_arrow_down(ax, 78, 15.5, 14)

# Emotion classification boxes
draw_box(ax, 2, 6, 16, 8,
         'ML Benchmark\n(n = 709)\n\nPDF retrieved: 217\nPDF pending: 492',
         color_incl, fontsize=8)

draw_box(ax, 21, 6, 16, 8,
         'Dataset Creation\n(n = 78)\n\nDatasets extracted: 36',
         color_incl, fontsize=8)

draw_box(ax, 40, 6, 16, 8,
         'Other\n(n = 968)\n\nNo benchmark\n(excluded)',
         color_excl, fontsize=8)

# Empathy classification boxes
draw_box(ax, 57, 6, 10, 8,
         'ML Papers\n(n = 138)\n\nExtracted:\n134',
         color_incl, fontsize=8)

draw_box(ax, 69, 6, 14, 8,
         'Dataset Papers\n(n = 13)\n\nNew datasets:\n11',
         color_incl, fontsize=8)

# Final Synthesis box
draw_box(ax, 85, 10, 13, 12,
         'SYNTHESIS\n\nDatasets:\n126\n\nEmotion: 105\nEmpathy: 21',
         '#E1BEE7', fontsize=8)

# Arrows to synthesis
draw_arrow_right(ax, 37, 85, 16)
draw_arrow_right(ax, 83, 85, 16)

# ============ LEGEND ============
legend_y = 1.5
legend_items = [
    (3, color_id, 'Identification'),
    (20, color_screen, 'Screening'),
    (37, color_elig, 'Eligibility'),
    (54, color_incl, 'Included'),
    (71, color_excl, 'Excluded')
]

for x, color, label in legend_items:
    ax.add_patch(FancyBboxPatch((x, legend_y), 4, 2.5,
                                facecolor=color, edgecolor=border_color, linewidth=1))
    ax.text(x + 5, legend_y + 1.25, label, fontsize=9, va='center')

plt.tight_layout()
plt.savefig('PRISMA_FLOW_DIAGRAM.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('PRISMA_FLOW_DIAGRAM.pdf', bbox_inches='tight', facecolor='white')
print("Saved: PRISMA_FLOW_DIAGRAM.png, PRISMA_FLOW_DIAGRAM.pdf")
