/* Any copyright is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/ */

#![allow(unused)]

use anyhow::Context;
use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
};

pub fn write_to_file<Q: AsRef<Path>, T: AsRef<[u8]>>(filepath: Q, data: T) -> anyhow::Result<()> {
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

pub fn get_example_filepath<Q: AsRef<Path>>(filename: Q) -> anyhow::Result<PathBuf> {
    let folder_pb: PathBuf = [env!("CARGO_MANIFEST_DIR"), "examples-output/"]
        .iter()
        .collect();
    if !std::fs::exists(&folder_pb)? {
        std::fs::create_dir(&folder_pb)?;
        log::info!("Created the folder 'examples-output'")
    }
    Ok(folder_pb.join(filename))
}

pub fn get_figure_filepath<Q: AsRef<Path>>(filename: Q) -> anyhow::Result<PathBuf> {
    let folder_pb: PathBuf = [env!("CARGO_MANIFEST_DIR"), "docs/figures/"]
        .iter()
        .collect();
    if !std::fs::exists(&folder_pb)? {
        let parent = folder_pb.parent().unwrap();
        if !std::fs::exists(parent)? {
            std::fs::create_dir(parent)?;
        }
        std::fs::create_dir(&folder_pb)?;
        log::info!("Created the folder 'docs/figures/'")
    }
    Ok(folder_pb.join(filename))
}

#[cfg(feature = "plot")]
pub fn save_scene<Q: AsRef<Path>>(
    scene: kuva::render::render::Scene,
    filepath: Q,
) -> anyhow::Result<()> {
    use kuva::prelude::*;

    let filepath = filepath.as_ref();

    let svg_data = SvgBackend.render_scene(&scene);

    write_to_file(filepath, svg_data)?;

    log::info!("Figure saved to the file at {}", filepath.display());
    Ok(())
}
