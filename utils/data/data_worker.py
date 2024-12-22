import numpy as np

from tqdm import tqdm


def load_scan(pcd_path):
    pcd_data = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 6)
    return pcd_data
    
def write_obj(points, out_filename):
    """Write points into `obj` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()

def write_obj_ply(points, out_filename):
    assert points.shape[1] in [3, 6]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if points.shape[1] == 6:
        header.extend([
            'property uchar red',
            'property uchar green',
            'property uchar blue',
        ])
    header.extend([
        'end_header',
    ])

    n_points, pcd_dim = points.shape
    with open(out_filename, 'w') as fout:
        for row in header:
            fout.write(row + '\n')
        for i in tqdm(range(n_points), leave=False):
            match pcd_dim:
                case 3:
                    fout.write(
                        '%f %f %f\n' %
                        (points[i, 0], points[i, 1], points[i, 2])
                    )
                case 6:
                    c = points[i, 3:].astype(int)
                    fout.write(
                        '%f %f %f %d %d %d\n' %
                        (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2])
                    )