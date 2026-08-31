from typing import Optional
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pyfar as pf
import pyarrow as pa
import pyarrow.parquet as pq

from math import pi
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pathlib import Path
from itertools import chain

SOUND_SPHERE_METADATA = """
Sound Sphere 2: A High-resolution HRTF Database"
AUTHORS:
  - Michaela Warnecke
  - Samuel Clapp
  - Zamir Ben-Hur
  - David Lou Alon
  - Sebastia V. Amengual Gari
  - Paul Calamia
LINK: https://facebookresearch.github.io/SS2_HRTF/
LICENSE: CC-BY-4.0
"""

def to_f32_audio(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)
    elif np.issubdtype(x.dtype, np.integer):
        print("→ not a floating-point subtype")
        info = np.iinfo(x.dtype)
        scale = max(abs(info.min), info.max)
        x = x.astype(np.float32) / scale
    else:
       raise ValueError(f"Input has the unexpected dtype of {x.dtype}")

    abs_max = np.abs(x).max().item()
    print(f"→ normalizing HRIRs (grouped) to ±1.0, had previous absmax of {abs_max:.2f}")
    x = x / abs_max

    return x

def process_sofa_file(filepath, metadata: Optional[str]) -> bytes:
    print(f"Working on '{filepath}'...")
    sofa_data: tuple[pf.Signal, pf.Coordinates, pf.Coordinates] = pf.io.read_sofa(filepath)
    hrirs, source_coordinates, receiver_coordinates = sofa_data
    hrir_samples = to_f32_audio(hrirs.time)

    buf = BytesIO()

    # sampling rate: f64
    print(f"→ sampling rate: {hrirs.sampling_rate:.2f}")
    buf.write(np.float64(hrirs.sampling_rate).tobytes(order="C"))

    # HRIR length (number of samples per HRIR): u64
    print(f"→ length per HRIR: {hrir_samples.shape[2]}")
    buf.write(hrir_samples.shape[2].to_bytes(length=8, byteorder="little", signed=False))

    # left ear position (cartesian): [f64; 3]
    left_ear_pos_cart = np.array([
        receiver_coordinates.x[0, 0],
        receiver_coordinates.y[0, 0],
        receiver_coordinates.z[0, 0]
    ], dtype=np.float64)
    print(f"→ left ear position: {left_ear_pos_cart}")
    buf.write(left_ear_pos_cart.tobytes(order="C"))

    # right ear position (cartesian): [f64; 3]
    right_ear_pos_cart = np.array([
        receiver_coordinates.x[1, 0],
        receiver_coordinates.y[1, 0],
        receiver_coordinates.z[1, 0]
    ], dtype=np.float64)
    print(f"→ right ear position: {right_ear_pos_cart}")
    buf.write(right_ear_pos_cart.tobytes(order="C"))

    # audio source radius: f64
    assert(np.allclose(source_coordinates.radius, source_coordinates.radius[0]))
    print(f"→ radius of the audio source: {source_coordinates.radius[0]:.2f}")
    buf.write(np.float64(source_coordinates.radius[0]).tobytes(order="C"))

    # number of HRIRs: u64
    n_elements = source_coordinates.cshape[0]
    buf.write(n_elements.to_bytes(length=8, byteorder="little", signed=False))

    for n in range(n_elements):
        # audio source position (shell): [f64; 2]
        src_pos = np.array([
            # See the pyfar docs for more info
            # https://pyfar.readthedocs.io/en/stable/classes/pyfar.coordinates.html
            source_coordinates.azimuth[n],
            pi - source_coordinates.colatitude[n]
        ], dtype=np.float64)
        buf.write(src_pos.tobytes(order="C"))

        # left HRIR: [f32; hrir_length]
        buf.write(hrir_samples[n, 0, :].tobytes(order="C"))

        # right HRIR: [f32; hrir_length]
        buf.write(hrir_samples[n, 1, :].tobytes(order="C"))

    # write metadata if it exists
    if metadata is not None:
        print("→ appended metadata to the end of the file")
        buf.write(metadata.encode(encoding="utf-8"))

    print(f"✓ finished processing {hrir_samples.shape[2]} HRIRs")

    buf.seek(0)
    return buf.read()


def single_file():
    #input_dirpath = Path("input/HRIRs_mannequins/")
    #input_filename = "HATS051123_1_processed.sofa"
    #input_filename = "KU100051023_1_processed.sofa"
    input_dirpath = Path("input/HRIRs/")
    input_filename = "ZTV406081722_1_processed.sofa"
    input_filepath = input_dirpath.joinpath(input_filename)

    bytes = process_sofa_file(input_filepath, SOUND_SPHERE_METADATA)

    output_dirpath = Path("output/")
    output_filename = input_filename[:-5] + ".hrir.asear"
    output_filepath = output_dirpath.joinpath(output_filename)

    with open(output_filepath, "wb") as f:
        f.write(bytes)


def all_files():
    output_dirpath = Path("output/")

    for input_dirpath, _, input_filenames in chain(Path("input/HRIRs/").walk(), Path("input/HRIRs_mannequins/").walk()):
        for input_filename in input_filenames:
            assert input_filename.endswith(".sofa")
            input_filepath = input_dirpath.joinpath(input_filename)

            table = process_sofa_file(input_filepath)

            output_filename = input_filename[:-5] + ".asear.hrtf.parquet"
            output_filepath = output_dirpath.joinpath(output_filename)
            print(f"Saving to {output_filepath}")
            pq.write_table(
                table,
                output_filepath,
                compression="zstd",
                compression_level=16,
            )

            return


def room():
    """ Run the following snippet if `SingleRoomSRIR_1.1` is not available
    import sofar as sf
    sf.update_conventions()
    """

    filepath = "input/4_IR_A.sofa"
    #filepath = "input/HRIRs/AKO536081622_1_processed.sofa"

    sofa_data: tuple[pf.Signal, pf.Coordinates, pf.Coordinates] = pf.io.read_sofa(filepath, verify=False)
    hrirs, source_coordinates, receiver_coordinates = sofa_data

    print(f"HRIRs: {hrirs.cshape}, source_coords: {source_coordinates.cshape}, recv_coords: {receiver_coordinates.cshape}")
    print()

    for i in range(source_coordinates.cshape[0]):
        print(f"src coord n°{i}:  ({float(source_coordinates.radius[i]):.1f}, {float(source_coordinates.azimuth[i]):.1f}, {float(source_coordinates.colatitude[i]):.1f})")
    print()

    for i in range(receiver_coordinates.cshape[0]):
        print(f"rcv coord n°{i}:  ({float(receiver_coordinates.radius[i,0]):.1f}, {float(receiver_coordinates.azimuth[i,0]):.1f}, {float(receiver_coordinates.colatitude[i,0]):.1f})")
    print()

    output: np.ndarray = to_f32_audio(hrirs._data[5][2])
    output_sr = hrirs._sampling_rate
    print("sampling rate:", output_sr)

    output_bytes = output.tobytes(order="C")
    with open("output/temp.raw", "wb") as f:
        f.write(output_bytes)

    print(output[0:100])


def plot_hrtf_points():
    input_dirpath = Path("input/HRIRs_mannequins/")
    input_filename = "HATS051123_1_processed.sofa"
    input_filepath = input_dirpath.joinpath(input_filename)

    sofa_data: tuple[pf.Signal, pf.Coordinates, pf.Coordinates] = pf.io.read_sofa(input_filepath)
    hrirs, source_coordinates, receiver_coordinates = sofa_data

    ax: Axes3D = source_coordinates.show()
    ax.scatter([0], [0], [0], marker='o', s=20)
    ax.set_box_aspect(None, zoom=0.85)
    ax.figure.savefig("plot.png", dpi=300, pad_inches=5.5)



if __name__ == "__main__":
    single_file()
