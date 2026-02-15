import numpy as np
import struct

def extract_fbin_subset(src_path, dst_path, topk, dim):
    with open(src_path, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        d = struct.unpack('i', f.read(4))[0]
        print(f"Total vectors in fbin: {n}, dim: {d}")
        assert d == dim

        # 读取前 topk 个向量
        data = np.fromfile(f, dtype=np.float32, count=topk * d).reshape(topk, d)

    with open(dst_path, 'wb') as fout:
        for i in range(topk):
            fout.write(struct.pack('i', d))
            fout.write(data[i].astype(np.float32).tobytes())
    print(f"Saved subset to {dst_path}")

# 用法示例
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    args = parser.parse_args()
    extract_fbin_subset(args.src, args.dst, topk=args.topk, dim=args.dim)