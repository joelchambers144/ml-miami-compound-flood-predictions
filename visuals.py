import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your data
df = pd.read_csv('results/flow_comparison_table.csv')

# 2. Create a figure and axis
# tight_layout helps remove unnecessary margins
fig, ax = plt.subplots(figsize=(10, 4)) 
ax.axis('off')  # Hide the axes (lines and labels)

# 3. Create the table
table = ax.table(
    cellText=df.values, 
    colLabels=df.columns, 
    cellLoc='center', 
    loc='center'
)

# 4. Optional: Styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Stretch the rows/columns for readability

# 5. Save as PNG
plt.savefig('table.png', bbox_inches='tight', dpi=300)
print("Table saved as table.png")