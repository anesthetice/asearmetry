from math import pi
import numpy as np
import pyfar as pf # ty: ignore
import pyarrow as pa # ty: ignore
import pyarrow.parquet as pq # ty: ignore


def to_f32_audio(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.integer):
        print("Not a floating-point subtype")
        info = np.iinfo(x.dtype)
        scale = max(abs(info.min), info.max)
        x = x.astype(np.float32) / scale
    else:
        x = x.astype(np.float32)
    np.clip(x, -1.0, 1.0, out=x)
    return x

def main():
    #fp = "./input/HRIRs_mannequins/KU100051023_1_processed.sofa"
    #fp = "./input/HRIRs/AKO536081622_1_processed.sofa"
    fp = "./input/HRIRs/SJN145081522_1_processed.sofa"

    print(f"Processing '{fp}'")
    sofa_data: tuple[pf.Signal, pf.Coordinates, pf.Coordinates] = pf.io.read_sofa(fp)
    hrirs, source_coordinates, receiver_coordinates = sofa_data

    hrir_samples = to_f32_audio(hrirs.time)

    assert hrir_samples.dtype == np.float32
    assert np.max(np.abs(hrir_samples)) <= 1.0

    nb_elements = source_coordinates.cshape[0]
    rows = []

    for n in range(nb_elements):
        rows.append({
            "src_radius": np.float32(source_coordinates.radius[n]),
            "src_azimuth": np.float32(source_coordinates.azimuth[n]),
            "src_zenith": np.float32(pi - source_coordinates.colatitude[n]),
            "hrir_left": hrir_samples[n, 0, :],
            "hrir_right": hrir_samples[n, 1, :],
        })

    schema = pa.schema(
        [
            ("src_radius", pa.float32()),
            ("src_azimuth", pa.float32()),
            ("src_zenith", pa.float32()),
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
            "sampling_rate": str(int(hrirs.sampling_rate)).encode(),
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

    table = pa.Table.from_pylist(rows, schema=schema)

    pq.write_table(
        table,
        "output/hrtf.parquet",
        compression="zstd",
        compression_level=9,
    )


if __name__ == "__main__":
    main()
