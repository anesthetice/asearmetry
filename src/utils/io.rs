/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

use anyhow::Context;
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::Path,
};

pub(crate) fn read_from_file<Q: AsRef<Path>>(
    filepath: Q,
    capacity: Option<usize>,
) -> anyhow::Result<Vec<u8>> {
    let filepath = filepath.as_ref();

    let mut file = OpenOptions::new()
        .read(true)
        .open(filepath)
        .with_context(|| format!("Failed to open the file with path {}", filepath.display()))?;

    let mut data: Vec<u8> = Vec::with_capacity(capacity.unwrap_or(2_usize.pow(16)));

    file.read_to_end(&mut data).with_context(|| {
        format!(
            "Failed to read the contents of the file with path {}",
            filepath.display()
        )
    })?;

    Ok(data)
}

pub(crate) fn write_to_file<Q: AsRef<Path>, T: AsRef<[u8]>>(
    filepath: Q,
    data: T,
) -> anyhow::Result<()> {
    let filepath = filepath.as_ref();

    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(filepath)
        .with_context(|| {
            format!(
                "Failed to open/create the file with path {}",
                filepath.display()
            )
        })?;

    file.write_all(data.as_ref()).with_context(|| {
        format!(
            "Failed to write the data to the file with path {}",
            filepath.display()
        )
    })?;

    file.sync_all()
        .with_context(|| format!("Failed to sync the file with path {}", filepath.display()))?;

    Ok(())
}
