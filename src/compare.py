import argparse
import numpy as np


def load_bem(path):
    data = np.loadtxt(path)
    return data[:, 0::2] + 1j * data[:, 1::2]


def main(bem_path, epgp_path):
    T_bem = load_bem(bem_path)
    T_epgp = np.load(epgp_path)

    assert T_bem.shape == T_epgp.shape, (T_bem.shape, T_epgp.shape)

    diff = T_epgp - T_bem
    rel = np.linalg.norm(diff) / np.linalg.norm(T_bem)

    print(f"shape: {T_bem.shape}")
    print(f"||T_bem||   = {np.linalg.norm(T_bem):.4f}")
    print(f"||T_epgp||  = {np.linalg.norm(T_epgp):.4f}")
    print(f"||T_epgp - T_bem|| / ||T_bem|| = {rel:.3e}")

    entrywise = np.abs(diff) / (np.abs(T_bem) + 1e-12)
    i, j = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    print(
        f"max abs diff at ({i},{j}): bem={T_bem[i, j]:+.4e}  epgp={T_epgp[i, j]:+.4e}"
    )
    print(f"median entrywise rel diff: {np.median(entrywise):.3e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bem", help="path to BEM T_matrix.dat")
    p.add_argument("epgp", help="path to EP-GP T_epgp.npy")
    args = p.parse_args()
    main(args.bem, args.epgp)
