import os
base = os.path.join(os.path.dirname(__file__), 'DATASET')
for split in ['TRAIN', 'TEST']:
    path = os.path.join(base, split)
    print('SPLIT', split)
    if not os.path.isdir(path):
        print('  missing', path)
        continue
    for cls in sorted(os.listdir(path)):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
            print(' ', cls, count)
        else:
            print(' ', cls, 'not-dir')
