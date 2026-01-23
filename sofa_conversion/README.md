Handling `.sofa` files is a bit of a pain unfortunately, there isn't currently a good rust solution,
furthermore it's cumbersome to use the `netcdf` crate as one of its dependencies uses version 1 of
an `HDF5` library...

My hacky solution is to just use python, specifically the `pyfar` library to convert any `.sofa` file into
a parquet database which can then be loaded with polars.

Although as this is quite a heavy solution, I might create another crate that depends on `asearmetry` to read
the parquet database and convert it into just a serialized struct.
