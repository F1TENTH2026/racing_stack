#!/usr/bin/env python3
"""<map>.pgm(nav2 점유격자) -> <map>_track.{pgm,yaml} (kiss_icp_localization용 트랙마스크).

kiss의 TrackMask는 픽셀값 >127을 "주행영역", 그 경계를 "벽(SDF zero-level)"으로 봅니다
(track_mask.hpp의 `px > 127 ? 1 : 0`). 그래서 cartographer 점유격자를 그대로 주면
미탐사(128)까지 주행영역이 되어 SDF가 트랙 밖으로 샙니다.

여기서는 nav2 규약(free_thresh)에 따라 **실제로 관측된 free 셀만** 시드로 삼아
flood-fill 하므로, 미탐사 영역이 자연스럽게 장벽 역할을 해서 트랙만 남습니다.

사용법:
    python3 make_track_mask.py <맵폴더>/<맵이름>.yaml
"""
import sys
import os

import numpy as np
from PIL import Image
from scipy import ndimage

# 이보다 작은 구멍만 메움 [px]. 0.05 m/px 기준 40px ≈ 0.1 m^2 — 스캔 노이즈 크기.
MAX_HOLE_PX = 40


def read_map_yaml(path):
    """map_server 스타일 yaml에서 필요한 키만 파싱 (yaml 의존성 없이)."""
    cfg = {'resolution': 0.05, 'origin': [0.0, 0.0, 0.0],
           'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.196}
    with open(path) as f:
        for line in f:
            line = line.split('#')[0]
            if ':' not in line:
                continue
            k, v = line.split(':', 1)
            k, v = k.strip(), v.strip()
            if k == 'image':
                cfg['image'] = v
            elif k in ('resolution', 'occupied_thresh', 'free_thresh'):
                cfg[k] = float(v)
            elif k == 'negate':
                cfg[k] = int(v)
            elif k == 'origin':
                cfg[k] = [float(x) for x in v.strip('[]').replace(',', ' ').split()]
    return cfg


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    yaml_path = os.path.abspath(sys.argv[1])
    map_dir = os.path.dirname(yaml_path)
    cfg = read_map_yaml(yaml_path)

    img_path = os.path.join(map_dir, cfg['image'])
    if not os.path.exists(img_path):           # yaml이 .png를 가리켜도 .pgm으로 폴백
        alt = os.path.splitext(img_path)[0] + '.pgm'
        img_path = alt if os.path.exists(alt) else img_path
    px = np.array(Image.open(img_path).convert('L'))

    # nav2 규약: occ = (255 - px)/255 (negate=0). occ < free_thresh 인 셀만 "관측된 free".
    # -> px > 255*(1 - free_thresh). 미탐사(128)는 여기서 탈락해 장벽 역할을 합니다.
    free_cut = 255.0 * (1.0 - cfg['free_thresh'])
    free = px > free_cut

    # 4-이웃 연결성: 대각선 틈으로 새는 것을 방지.
    labels, n = ndimage.label(free, structure=ndimage.generate_binary_structure(2, 1))
    if n == 0:
        print('ERROR: free 영역이 없습니다 — free_thresh / 맵을 확인하세요.')
        return 1

    sizes = ndimage.sum(free, labels, range(1, n + 1))
    track = labels == (int(np.argmax(sizes)) + 1)   # 가장 큰 연결영역 = 트랙

    # 스캔 노이즈로 생긴 '작은' 구멍만 메움. 트랙은 보통 도넛 형태라 전체 구멍메우기
    # (binary_fill_holes)를 쓰면 가운데 섬까지 주행영역이 되므로 면적으로 걸러냅니다.
    track = ndimage.binary_closing(track, structure=np.ones((3, 3)))
    holes = ndimage.binary_fill_holes(track) & ~track
    hlab, hn = ndimage.label(holes)
    if hn:
        hsz = ndimage.sum(holes, hlab, range(1, hn + 1))
        small = np.isin(hlab, [i + 1 for i, s in enumerate(hsz) if s <= MAX_HOLE_PX])
        track = track | small

    out_stem = os.path.splitext(yaml_path)[0] + '_track'
    out_pgm, out_yaml = out_stem + '.pgm', out_stem + '.yaml'

    Image.fromarray(np.where(track, 255, 0).astype(np.uint8), mode='L').save(out_pgm)
    with open(out_yaml, 'w') as f:
        f.write(f"# kiss_icp_localization 2D scan-to-SDF용 트랙마스크.\n"
                f"# {os.path.basename(yaml_path)}에서 make_track_mask.py로 생성 (수정 금지, 재생성할 것).\n"
                f"# 트랙=255(주행영역), 벽/트랙밖=0. 경계가 SDF의 zero-level(=벽)이 됩니다.\n"
                f"image: {os.path.basename(out_pgm)}\n"
                f"resolution: {cfg['resolution']:.6f}\n"
                f"origin: [{cfg['origin'][0]}, {cfg['origin'][1]}, {cfg['origin'][2]}]\n"
                f"negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")

    total = track.size
    print(f'입력 : {img_path}  ({px.shape[1]}x{px.shape[0]})')
    print(f'출력 : {out_pgm}\n       {out_yaml}')
    print(f'연결영역 {n}개 중 최대 영역 채택')
    print(f'트랙(255) : {track.sum():6d} px ({track.sum()/total*100:.1f}%)')
    print(f'그외 (0)  : {total-track.sum():6d} px ({(total-track.sum())/total*100:.1f}%)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
