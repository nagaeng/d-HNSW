# convert.py
import numpy as np
import struct
import sys

def convert_fbin_to_fvecs(fbin_path, fvecs_path):
    with open(fbin_path, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        d = struct.unpack('i', f.read(4))[0]
        print(f"Converting {n} vectors of dimension {d}...")

        data = np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)

    with open(fvecs_path, 'wb') as fout:
        for i in range(n):
            fout.write(struct.pack('i', d))
            fout.write(data[i].astype(np.float32).tobytes())

    print(f"Saved to {fvecs_path}")

def convert_ibin_to_ivecs(input_path, output_path):
    with open(input_path, 'rb') as f:
        num_vectors = int(np.fromfile(f, dtype='int32', count=1)[0])
        dim = int(np.fromfile(f, dtype='int32', count=1)[0])
        print(f"Converting {num_vectors} vectors with dimension {dim}...")

        data = np.fromfile(f, dtype='int32').reshape(num_vectors, dim)

    with open(output_path, 'wb') as f_out:
        for i in range(num_vectors):
            f_out.write(np.int32(dim).tobytes())
            f_out.write(data[i].tobytes())

    print(f"Saved to {output_path}")
def convert_fbin_to_fvecs_streaming(fbin_path, fvecs_path, chunk_size=10000):
    with open(fbin_path, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        d = struct.unpack('i', f.read(4))[0]
        print(f"Converting {n} vectors of dimension {d}...")

        with open(fvecs_path, 'wb') as fout:
            num_chunks = (n + chunk_size - 1) // chunk_size
            for chunk_idx in range(num_chunks):
                remaining = n - chunk_idx * chunk_size
                cur_chunk_size = min(chunk_size, remaining)

                data = np.fromfile(f, dtype=np.float32, count=cur_chunk_size * d)
                if data.size != cur_chunk_size * d:
                    raise ValueError(f"Unexpected EOF at chunk {chunk_idx}")

                data = data.reshape(cur_chunk_size, d)
                for i in range(cur_chunk_size):
                    fout.write(struct.pack('i', d))
                    fout.write(data[i].astype(np.float32).tobytes())

                print(f"Chunk {chunk_idx + 1}/{num_chunks} done")
if __name__ == "__main__":
    assert len(sys.argv) == 4
    mode = sys.argv[1]  # "fbin" or "ibin"
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    if mode == "fbin":
        convert_fbin_to_fvecs(input_path, output_path)
    elif mode == "ibin":
        convert_ibin_to_ivecs(input_path, output_path)
    elif mode == "fbin_streaming":
        convert_fbin_to_fvecs_streaming(input_path, output_path)
    else:
        raise ValueError("Unknown mode. Use 'fbin' or 'ibin'.")