import os
import re

def wrap_main(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    content = "".join(lines)
    if 'if __name__ == "__main__":' in content or "if __name__ == '__main__':" in content:
        return # Already wrapped

    new_lines = []
    found_loop = False
    
    # Simple heuristic: find where history/rounds/for loops start
    for i, line in enumerate(lines):
        if not found_loop and (line.strip().startswith('history = []') or line.strip().startswith('for r in')):
            new_lines.append('if __name__ == "__main__":\n')
            found_loop = True
        
        if found_loop:
            new_lines.append('    ' + line)
        else:
            new_lines.append(line)
            
    if found_loop:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Wrapped: {filepath}")

for root, dirs, files in os.walk('.'):
    if 'backups_original' in root or 'venv' in root: continue
    for file in files:
        if file.endswith('.py'):
            wrap_main(os.path.join(root, file))
