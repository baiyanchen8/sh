import os
import re

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 1. Strip all transforms.Normalize inside transforms.Compose
    content = re.sub(r'(?m)^\s*transforms\.Normalize\(IMAGENET_MEAN,\s*IMAGENET_STD\),?\s*\r?\n', '', content)

    # 2. Ensure NORMALIZER is defined
    if 'NORMALIZER =' not in content:
        # Place it after transform_test
        content = re.sub(r'(transform_test = transforms\.Compose\(\[.*?\]\))', r'\1\n\nNORMALIZER = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)\n', content, flags=re.DOTALL)

    # 3. For any "x, y = x.to(DEVICE), y.to(DEVICE)" we append "x = NORMALIZER(x)"
    # We use a lambda to get the exact indentation of the matched line
    def insert_norm_x(match):
        indent = match.group(1)
        full_match = match.group(0)
        # Avoid double insertion
        if 'NORMALIZER' in content[match.end():match.end()+40]:
            return full_match
        return f"{full_match}\n{indent}x = NORMALIZER(x)"

    content = re.sub(r'^([ \t]*)(x,\s*y\s*=\s*x\.to\(DEVICE\),\s*y\.to\(DEVICE\))', insert_norm_x, content, flags=re.MULTILINE)

    # 4. Same for "xt = torch.clamp(...)" or "xt = torch.stack(...)"
    def insert_norm_xt(match):
        indent = match.group(1)
        full_match = match.group(0)
        if 'NORMALIZER' in content[match.end():match.end()+40]:
            return full_match
        return f"{full_match}\n{indent}xt = NORMALIZER(xt)"
        
    content = re.sub(r'^([ \t]*)(xt\s*=\s*torch\.clamp\(xb \+ trig, 0, 1\))', insert_norm_xt, content, flags=re.MULTILINE)
    content = re.sub(r'^([ \t]*)(xt\s*=\s*torch\.stack\(\[add_trigger\(img\) for img in x(?:b)?\]\)\.to\(DEVICE\))', insert_norm_xt, content, flags=re.MULTILINE)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Patched: {filepath}")

for root, dirs, files in os.walk('.'):
    if 'backups_original' in root: continue
    for file in files:
        if file.endswith('.py') and ('cifar' in file or 'mnist' in file):
            patch_file(os.path.join(root, file))

