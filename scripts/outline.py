"""
This generates a table of contents that you can use at the top of your notebook.
# How to use

- edit this script in your editor f.e. Spyder
- paste the location of your notebook into the notebook variable
- run the code
- the table of content gets printed
- copy paste the table of content into your notebook.
"""

import json
import sys

notebook =  sys.argv[1]

with open(notebook) as f:
    data = json.load(f)

cells = data['cells']
cells = [n for n in cells if n.get('cell_type','') == 'markdown']

def check_title(line):
    return '#' in line

titles = []
for cell in cells:
    line = cell.get('source',[])[0]
    if check_title(line):
        titles.append(line)

def fix_title_1(title):
    title = title.replace(">","")
    return title.replace("#","").strip()

def fix_title_2(title):
    title = title.replace(">","").strip()
    title = title.replace("# ", "#")
    title = title.replace("###", "#")
    title = title.replace("##", "#")
    title = title.replace(" ", "-")
    return title

def tabs(title):
    return "    " * (title.count("#")-2)

titles = ["{}- [{}]({})".format(tabs(t), fix_title_1(t), fix_title_2(t)) for t in titles]

for title in titles:
    print(title)
