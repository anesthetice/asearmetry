use std::iter::chain;

use approx::assert_abs_diff_eq;
use itertools::Itertools;
use rstar::{RTree, primitives::GeomWithData};

use crate::{
    Hertz, Meters, Radians, Seconds,
    coordinates::{Cart3D, Shell2D, Sphere3D},
    signal::{
        AudioBuffer, AudioSignalConvolution, AudioSignalCore, ChannelBuffers, ChannelBuffersSlice,
        MonoAudioBuf, StereoAudioBuf,
    },
    trajectory::Trajectory,
};

pub type HrirProjection = GeomWithData<Shell2D, ChannelBuffers<2>>;

pub struct Binauralizer {
    pub hrir_tree: RTree<HrirProjection>,
    pub hrir_radius: Meters,
    pub hrir_size: usize,
    pub left_ear_pos: Cart3D,
    pub right_ear_pos: Cart3D,
    pub hrir_sample_rate: Hertz,
}

impl Binauralizer {
    pub fn run<T>(&self, src: &MonoAudioBuf, trajectory: Trajectory<T>) -> StereoAudioBuf
    where
        T: Into<Cart3D> + Copy,
    {
        let src_sample_rate = src.sample_rate();
        assert_eq!(self.hrir_sample_rate as u32, src_sample_rate as u32);

        let block_size = src_sample_rate as usize / 10;
        let time_size: Seconds = 1.0 / 10.0;
        assert!(time_size > trajectory.δt);

        let to_traj_index =
            |idx: usize| ((time_size / trajectory.δt) * idx as f32).floor() as usize;

        let (blocks, rem) = src.as_blocks(block_size);

        let blocks_len = blocks.len();

        let iter = blocks
            .into_iter()
            .enumerate()
            .map(|(i, block)| {
                let hrir = self.get_hrir(*trajectory.path.get(to_traj_index(i)).unwrap());
                block.convolve(&hrir)
            })
            .chain(rem.map(|block| {
                let hrir = self.get_hrir(*trajectory.path.get(to_traj_index(blocks_len)).unwrap());
                block.convolve(&hrir)
            }));

        ChannelBuffers::<2>::crossfade_concatenate(iter, self.hrir_size * 1)
            .into_audio_buf(src_sample_rate)
    }

    /// Accounts for parallax
    pub fn get_hrir<'a>(&'a self, pos: impl Into<Cart3D>) -> ChannelBuffersSlice<'a, 2> {
        let pos = pos.into();

        println!("-> cart: {pos:?}   sphe: {:?}", Sphere3D::from(pos));

        let inner_compute = |ear: Cart3D| {
            let slope_xy = (pos.y - ear.y) / (pos.x - ear.x);
            let ic_y = ear.y - slope_xy * ear.x;
            println!("y = {slope_xy:.2} * x + {ic_y:.2}");

            let slope_xz = (pos.z - ear.z) / (pos.x - ear.x);
            let ic_z = ear.z - slope_xz * ear.x;
            println!("z = {slope_xz:.2} * x + {ic_z:.2}");

            let a = 1.0 + slope_xy.powi(2) + slope_xz.powi(2);
            let b = 2.0 * slope_xy * ic_y + 2.0 * slope_xz * ic_z;
            let c = ic_y.powi(2) + ic_z.powi(2) - self.hrir_radius.powi(2);

            let x1 = (-b + (b.powi(2) - 4.0 * a * c).sqrt()) / (2.0 * a);
            let xyz_1 = Cart3D::new(x1, (slope_xy * x1) + ic_y, (slope_xz * x1) + ic_z);

            let x2 = (-b - (b.powi(2) - 4.0 * a * c).sqrt()) / (2.0 * a);
            let xyz_2 = Cart3D::new(x2, slope_xy * x2 + ic_y, slope_xz * x2 + ic_z);

            println!("1: {xyz_1:?};   2: {xyz_2:?}");

            let query: Shell2D = if pos.dist(xyz_1) < pos.dist(xyz_2) {
                xyz_1.into()
            } else {
                xyz_2.into()
            };

            println!("{query:?}");

            self.hrir_tree
                .nearest_neighbor(&query.into())
                .unwrap()
                .data
                .as_ref()
        };

        ChannelBuffersSlice([
            inner_compute(self.left_ear_pos).0[0],
            inner_compute(self.right_ear_pos).0[1],
        ])
    }
}
