#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick manifest/mask sanity audit for building segmentation experiments."""
import argparse, os
from pathlib import Path
import numpy as np
from PIL import Image


def normalize(p):
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(p))))


def candidate_mask_paths(image_path: str):
    p = Path(image_path)
    suffixes = [p.suffix] if p.suffix else []
    suffixes += [".png", ".tif", ".tiff", ".jpg", ".jpeg"]
    seen, out = set(), []
    reps = [
        ("/images/", "/dual_channel_labels/"), ("\\images\\", "\\dual_channel_labels\\"),
        ("/images/", "/labels/"), ("\\images\\", "\\labels\\"),
        ("/JPEGImages/", "/SegmentationClass/"), ("\\JPEGImages\\", "\\SegmentationClass\\"),
        ("/imgs/", "/masks/"), ("\\imgs\\", "\\masks\\"),
    ]
    s = str(p)
    for old, new in reps:
        if old in s:
            base = str(Path(s.replace(old, new)).with_suffix(""))
            for suf in suffixes:
                out.append(base + suf)
    for folder in ["dual_channel_labels", "labels", "masks", "mask", "gt", "GT"]:
        base = str(p.parent.parent / folder / p.stem)
        for suf in suffixes:
            out.append(base + suf)
    clean = []
    for x in out:
        x = os.path.normpath(x)
        if x not in seen and os.path.exists(x) and normalize(x) != normalize(image_path):
            seen.add(x); clean.append(x)
    return clean


def parse_line(line):
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    parts = [x.strip() for x in (line.split(',') if ',' in line else line.split()) if x.strip()]
    return (parts[0], parts[1] if len(parts) > 1 else None)


def read_manifest(path, project_root):
    items=[]
    for line in open(path, encoding='utf-8'):
        z=parse_line(line)
        if z is None: continue
        img, mask=z
        if project_root and not os.path.isabs(img): img=str(Path(project_root)/img)
        if mask and project_root and not os.path.isabs(mask): mask=str(Path(project_root)/mask)
        if mask is None:
            cands=candidate_mask_paths(img)
            mask=cands[0] if cands else None
        items.append((normalize(img), normalize(mask) if mask else None))
    return items


def mask_stats(mask_path, mask_channel='auto', auto_invert=True):
    arr=np.array(Image.open(mask_path))
    if arr.ndim == 2:
        raw_unique=len(np.unique(arr)); candidates=[('gray', arr)]
    elif arr.ndim == 3:
        raw_unique=len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))
        if mask_channel != 'auto':
            candidates=[(str(mask_channel), arr[..., int(mask_channel)])]
        elif arr.shape[-1] >= 3 and np.array_equal(arr[...,0], arr[...,1]) and np.array_equal(arr[...,1], arr[...,2]):
            candidates=[('rgb_gray', arr[...,0])]
        else:
            candidates=[(str(i), arr[...,i]) for i in range(arr.shape[-1])]
    else:
        raise RuntimeError(f'unsupported mask shape {arr.shape}')
    vals=[]
    for name,ch in candidates:
        b=(ch>0).astype(np.uint8)
        fg=float(b.mean())
        vals.append((name,fg,b))
    plausible=[v for v in vals if 0.001 <= v[1] <= 0.80]
    name,fg,b = max(plausible, key=lambda x:x[1]) if plausible else min(vals, key=lambda x:abs(x[1]-0.2))
    inv=False
    if auto_invert and fg > 0.80:
        ib=1-b; ifg=float(ib.mean())
        if 0.001 <= ifg <= 0.80:
            b=ib; fg=ifg; inv=True
    return dict(shape=tuple(arr.shape), raw_unique=int(raw_unique), chosen_channel=name, fg=float(fg), inverted=inv)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project_root', default='./data')
    ap.add_argument('--manifest_dir', default=None)
    ap.add_argument('--manifests', nargs='+', default=['source_train.txt','source_val.txt','target_val.txt','target_final_test_8402.txt','target_pilot_test_500.txt'])
    ap.add_argument('--samples', type=int, default=30)
    ap.add_argument('--mask_channel', default='auto')
    ap.add_argument('--no_auto_invert', action='store_true')
    args=ap.parse_args()
    root=Path(args.project_root).resolve()
    mdir=Path(args.manifest_dir).resolve() if args.manifest_dir else root/'data/splits/e0_manifest'
    print('[AUDIT] root=', root)
    print('[AUDIT] manifest_dir=', mdir)
    for mf in args.manifests:
        path=mdir/mf
        items=read_manifest(path, root)
        idxs=np.linspace(0, len(items)-1, min(args.samples,len(items)), dtype=int).tolist()
        ratios=[]; missing=0; many_unique=0; dense=0; empty=0
        print('\n===', mf, 'n=', len(items), 'sample=', len(idxs), '===')
        for k,idx in enumerate(idxs):
            img,mask=items[idx]
            if not mask or not os.path.exists(mask):
                missing += 1; print('MISSING MASK', idx, img, mask); continue
            st=mask_stats(mask, args.mask_channel, not args.no_auto_invert)
            ratios.append(st['fg']); many_unique += int(st['raw_unique']>32); dense += int(st['fg']>0.8); empty += int(st['fg']<1e-6)
            if k < 5:
                print(f'[{idx}] img={img}')
                print(f'[{idx}] mask={mask} stats={st}')
        if ratios:
            print(f'SUMMARY {mf}: fg_mean={np.mean(ratios):.4f}, fg_min={np.min(ratios):.4f}, fg_max={np.max(ratios):.4f}, empty={empty}, dense>0.8={dense}, raw_unique>32={many_unique}, missing={missing}')
            if np.mean(ratios)>0.8 or many_unique>0:
                print('!!! SUSPICIOUS: check whether masks are inverted, wrong-channel, or image files were read as masks.')

if __name__ == '__main__':
    main()
