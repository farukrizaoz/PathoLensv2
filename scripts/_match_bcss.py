import urllib.request, csv
from pathlib import Path

url = 'https://raw.githubusercontent.com/PathologyDataScience/BCSS/master/meta/slide_magnifications.csv'
with urllib.request.urlopen(url, timeout=15) as r:
    lines = r.read().decode().strip().splitlines()

bcss_barcodes = {}
for line in lines[1:]:
    parts = line.split(',')
    if len(parts) >= 2:
        barcode = parts[0].strip()
        svs_name = parts[1].strip()
        bcss_barcodes[barcode] = svs_name

print(f'BCSS slides: {len(bcss_barcodes)}')

manifest = []
with open('data/metadata/gdc_manifest.txt') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        manifest.append(row)

matched = []
unmatched = []
for barcode in bcss_barcodes:
    found = None
    for row in manifest:
        if barcode in row.get('filename', ''):
            found = row
            break
    if found:
        matched.append(found)
    else:
        unmatched.append(barcode)

print(f'Matched: {len(matched)}/151')
if unmatched:
    print(f'Unmatched ({len(unmatched)}): {unmatched[:5]}')

Path('data/metadata').mkdir(parents=True, exist_ok=True)
with open('data/metadata/gdc_manifest_bcss151.txt', 'w') as f:
    f.write('id\tfilename\tmd5\tsize\tstate\n')
    for row in matched:
        f.write(f"{row['id']}\t{row['filename']}\t{row.get('md5','')}\t{row.get('size','')}\t{row.get('state','')}\n")

with open('data/metadata/bcss_slide_ids.txt', 'w') as f:
    for row in matched:
        f.write(row['filename'].replace('.svs','') + '\n')

total_gb = sum(int(r.get('size',0) or 0) for r in matched) / 1e9
print(f'Filtered manifest: {len(matched)} slides, {total_gb:.1f} GB')
print('Saved: data/metadata/gdc_manifest_bcss151.txt')
