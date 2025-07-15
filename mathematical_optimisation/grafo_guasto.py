import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Crea figura con due subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Funzione per disegnare un nodo
def draw_node(ax, x, y, label, color='lightblue'):
    circle = plt.Circle((x, y), 0.3, color=color, ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold')

# Funzione per disegnare un link
def draw_link(ax, x1, y1, x2, y2, color='black', width=2, style='-'):
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, linestyle=style)

# Funzione per disegnare una freccia curva
def draw_curved_arrow(ax, x1, y1, x2, y2, color='red', label=''):
    # Calcola punto di controllo per la curva
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2 - 0.5
    
    # Crea percorso curvo
    t = np.linspace(0, 1, 100)
    x = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
    y = (1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * y2
    
    ax.plot(x, y, color=color, linewidth=2)
    
    # Aggiungi freccia
    ax.annotate('', xy=(x2, y2), xytext=(x[-2], y[-2]),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Aggiungi etichetta
    if label:
        ax.text(mid_x, mid_y, label, ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# --- Scenario Normale (ax1) ---
ax1.set_xlim(-1, 7)
ax1.set_ylim(-1, 3)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Normal Scenary', fontsize=16, fontweight='bold', pad=20)

# Disegna nodi
nodes_normal = {'A': (0, 1), 'B': (2, 1), 'C': (4, 1), 'D': (6, 1)}
for node, (x, y) in nodes_normal.items():
    draw_node(ax1, x, y, node)

# Disegna link
draw_link(ax1, 0.3, 1, 1.7, 1, 'green', 3)
draw_link(ax1, 2.3, 1, 3.7, 1, 'green', 3)
draw_link(ax1, 4.3, 1, 5.7, 1, 'green', 3)

# Aggiungi frecce per il flusso
ax1.annotate('', xy=(1.5, 1.3), xytext=(0.5, 1.3),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax1.annotate('', xy=(3.5, 1.3), xytext=(2.5, 1.3),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax1.annotate('', xy=(5.5, 1.3), xytext=(4.5, 1.3),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))

# Etichetta flusso
ax1.text(3, 2, 'Flow: A → B → C → D', ha='center', fontsize=12,
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))

# Evidenzia link B-C
rect = Rectangle((2.3, 0.7), 1.4, 0.6, linewidth=2, edgecolor='blue', 
                 facecolor='none', linestyle='--')
ax1.add_patch(rect)
ax1.text(3, 0.3, 'Link B-C', ha='center', fontsize=10, color='blue')

# --- Scenario di Guasto (ax2) ---
ax2.set_xlim(-1, 7)
ax2.set_ylim(-2, 3)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('Scenary with broken link B-C', fontsize=16, fontweight='bold', pad=20)

# Disegna nodi principali
for node, (x, y) in nodes_normal.items():
    draw_node(ax2, x, y, node)

# Disegna nodi aggiuntivi per il percorso alternativo
draw_node(ax2, 2, -1, 'E', 'lightyellow')
draw_node(ax2, 4, -1, 'F', 'lightyellow')

# Disegna link funzionanti
draw_link(ax2, 0.3, 1, 1.7, 1, 'green', 3)
draw_link(ax2, 4.3, 1, 5.7, 1, 'green', 3)

# Disegna link guasto con X
draw_link(ax2, 2.3, 1, 3.7, 1, 'red', 3, '--')
ax2.text(3, 1, '✗', ha='center', va='center', fontsize=24, color='red', fontweight='bold')

# Disegna percorso alternativo
draw_link(ax2, 2, 0.7, 2, -0.7, 'orange', 2)
draw_link(ax2, 2.3, -1, 3.7, -1, 'orange', 2)
draw_link(ax2, 4, -0.7, 4, 0.7, 'orange', 2)

# Frecce per il nuovo percorso
ax2.annotate('', xy=(2, -0.5), xytext=(2, 0.5),
             arrowprops=dict(arrowstyle='->', color='orange', lw=2))
ax2.annotate('', xy=(3.5, -0.8), xytext=(2.5, -0.8),
             arrowprops=dict(arrowstyle='->', color='orange', lw=2))
ax2.annotate('', xy=(4, 0.5), xytext=(4, -0.5),
             arrowprops=dict(arrowstyle='->', color='orange', lw=2))

# Frecce per il flusso principale
ax2.annotate('', xy=(1.5, 1.3), xytext=(0.5, 1.3),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax2.annotate('', xy=(5.5, 1.3), xytext=(4.5, 1.3),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))

# Etichette
ax2.text(3, 2, 'Flow: A → B → (E → F) → C → D', ha='center', fontsize=12,
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

ax2.text(3, -1.7, 'Backup path\nof IGP', ha='center', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.3))

# Aggiungi annotazione per il meccanismo
ax2.text(1, -0.5, 'Automatic\nrerouting', ha='center', fontsize=9, style='italic')

# Titolo generale
fig.suptitle('Link Restoration: Example of Recovery from Failure', fontsize=18, fontweight='bold')

# Aggiungi legenda
legend_elements = [
    plt.Line2D([0], [0], color='green', lw=3, label='active path'),
    plt.Line2D([0], [0], color='red', lw=3, linestyle='--', label='broken link'),
    plt.Line2D([0], [0], color='orange', lw=2, label='backup path of IGP'),
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('link_restoration_example.png', dpi=300, bbox_inches='tight')
plt.show()