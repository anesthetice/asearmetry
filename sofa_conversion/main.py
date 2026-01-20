import pyfar as pf # ty: ignore
import pyarrow as pa # ty: ignore
import pyarrow.parquet as pq # ty: ignore
from math import pi


def main():
    sofa_data: tuple[pf.Signal, pf.Coordinates, pf.Coordinates] = pf.io.read_sofa("input/HRIRs_mannequins/KU100051023_1_processed.sofa")
    hrirs, source_coordinates, receiver_coordinates = sofa_data

    nb_elements = source_coordinates.cshape[0]

    rows = []
    for n in range(nb_elements):
        rows.append({
            "src_radius": float(source_coordinates.radius[n]),
            "src_azimuth": float(source_coordinates.azimuth[n]),
            "src_zenith": float(pi - source_coordinates.colatitude[n]),
            "hrir_left": hrirs.time[n, 0, :],
            "hrir_right": hrirs.time[n, 1, :],
        })
    table = pa.Table.from_pylist(rows)

    metadata = {
        "ORIGIN": b"Sound Sphere 2: A High-resolution HRTF Database",
        "AUTHORS": b"Michaela Warnecke, Samuel Clapp, Zamir Ben-Hur, David Lou Alon, Sebastia V. Amengual Gari, and Paul Calamia",
        "LINK": b"https://facebookresearch.github.io/SS2_HRTF/",
        "LICENSE": b"CC-BY-4.0",
        "sampling_rate": str(hrirs.sampling_rate).encode(),
        "left_ear_position_cartesian": f"{receiver_coordinates.x[0]}, {receiver_coordinates.y[0]}, {receiver_coordinates.z[0]}".encode(),
        "right_ear_position_cartesian": f"{receiver_coordinates.x[1]}, {receiver_coordinates.y[1]}, {receiver_coordinates.z[1]}".encode(),
    }

    schema = table.schema.with_metadata(metadata)
    table = table.cast(schema)

    pq.write_table(
        table,
        "hrtf.parquet",
        compression="zstd",
        compression_level=9,
    )


if __name__ == "__main__":
    main()
