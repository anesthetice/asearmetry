use itertools::Itertools;
use rstar::{RTree, primitives::GeomWithData};

use crate::{
    Radians,
    coordinates::{Shell2D, Sphere3D},
    signal::{ChannelBuffers, StereoAudioBuf},
};

pub type HrirProjection = GeomWithData<Shell2D, ChannelBuffers<2>>;

pub struct Binauralizer {
    pub hrir_tree: RTree<HrirProjection>,
    pub hrir_sample_rate: f32,
}

impl Binauralizer {
    pub fn new<T: Into<Sphere3D>>(data: Vec<(T, StereoAudioBuf)>) -> Self {
        let hrir_sample_rate = data.first().unwrap().1.sample_rate();
        let hrir_tree = RTree::bulk_load(
            data.into_iter()
                .map(|(pos, stereo_buf)| {
                    let pos = pos.into();
                    if stereo_buf.sample_rate != hrir_sample_rate {
                        panic!("Sample rate mismatch between HRIRs");
                    }
                    HrirProjection::new(Shell2D::new(pos.θ, pos.φ), stereo_buf.into())
                })
                .collect_vec(),
        );
        Self {
            hrir_tree,
            hrir_sample_rate,
        }
    }
}
