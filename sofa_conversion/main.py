import numpy as np
import pyfar as pf
import pyarrow as pa
import pyarrow.parquet as pq

from math import pi
from pathlib import Path
from itertools import chain

def to_f32_audio(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)
        abs_max = np.abs(x).max().item()
        if abs_max > 1.0:
            print(f"Warning, got absmax of {abs_max}, which is greater than 1.0, normalizing all hrirs together")
            x = x / abs_max
    elif np.issubdtype(x.dtype, np.integer):
        print("Not a floating-point subtype")
        info = np.iinfo(x.dtype)
        scale = max(abs(info.min), info.max)
        x = x.astype(np.float32) / scale
    else:
       raise ValueError(f"Input has the unexpected dtype of {x.dtype}")

    return x

def process_sofa_file(filepath) -> pa.Table:
    print(f"Processing '{filepath}'")
    sofa_data: tuple[pf.Signal, pf.Coordinates, pf.Coordinates] = pf.io.read_sofa(filepath)
    hrirs, source_coordinates, receiver_coordinates = sofa_data

    hrir_samples = to_f32_audio(hrirs.time)

    nb_elements = source_coordinates.cshape[0]
    rows = []

    for n in range(nb_elements):
        rows.append({
            "src_radius": np.float64(source_coordinates.radius[n]),
            "src_azimuth": np.float64(source_coordinates.azimuth[n]),
            "src_zenith": np.float64(pi - source_coordinates.colatitude[n]),
            "hrir_left": hrir_samples[n, 0, :],
            "hrir_right": hrir_samples[n, 1, :],
        })

    schema = pa.schema(
        [
            ("src_radius", pa.float64()),
            ("src_azimuth", pa.float64()),
            ("src_zenith", pa.float64()),
            ("hrir_left", pa.list_(pa.float32())),
            ("hrir_right", pa.list_(pa.float32())),
        ],
        metadata={
            "ORIGIN": b"Sound Sphere 2: A High-resolution HRTF Database",
            "AUTHORS":
                b"Michaela Warnecke, Samuel Clapp, Zamir Ben-Hur, "
                b"David Lou Alon, Sebastia V. Amengual Gari, Paul Calamia",
            "LINK": b"https://facebookresearch.github.io/SS2_HRTF/",
            "LICENSE": b"CC-BY-4.0",
            "sampling_rate": str(hrirs.sampling_rate).encode(),
            "left_ear_position_cartesian":
                f"{receiver_coordinates.x[0, 0]}, "
                f"{receiver_coordinates.y[0, 0]}, "
                f"{receiver_coordinates.z[0, 0]}".encode(),
            "right_ear_position_cartesian":
                f"{receiver_coordinates.x[1, 0]}, "
                f"{receiver_coordinates.y[1, 0]}, "
                f"{receiver_coordinates.z[1, 0]}".encode(),
        },
    )

    return pa.Table.from_pylist(rows, schema=schema)


def main2():
    output_dirpath = Path("output/")

    for input_dirpath, _, input_filenames in chain(Path("input/HRIRs/").walk(), Path("input/HRIRs_mannequins/").walk()):
        for input_filename in input_filenames:
            assert input_filename.endswith(".sofa")
            input_filepath =  input_dirpath.joinpath(input_filename)

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


def main():
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





if __name__ == "__main__":
    main()
