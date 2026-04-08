// Modules
#[cfg(feature = "polars")]
mod load;

use std::{f32, num::FpCategory};

// Imports
use crate::{
    Meters,
    audio::{
        AudioBuffer, AudioBufferSlice, AudioSignal, DiscreteSignal, DiscreteSignalUtils,
        StereoAudioBuf,
    },
    coordinates::{Cart3D, Shell2D},
    trajectory::Trajectory,
};
use rstar::{RTree, primitives::GeomWithData};

pub type HrirProjection = GeomWithData<Shell2D, AudioBuffer<2>>;

pub struct Binauralizer {
    pub hrir_tree: RTree<HrirProjection>,
    pub hrir_radius: Meters,
    pub hrir_size: usize,
    pub hrir_sampling_rate: u32,
    pub left_ear_pos: Cart3D,
    pub right_ear_pos: Cart3D,
}

impl Binauralizer {
    pub fn run<A, T>(&self, src: A, trajectory: Trajectory<T>) -> StereoAudioBuf
    where
        A: AudioSignal<1>,
        T: Into<Cart3D> + Copy,
    {
        let src_sampling_rate = src.sampling_rate().expect("No sampling rate");
        assert_eq!(self.hrir_sampling_rate, src_sampling_rate);
        let sampling_rate = src_sampling_rate as f32;

        let final_position = *trajectory.path.last().unwrap();
        let mut position_with_src_end_idx = trajectory
            .path
            .into_iter()
            .enumerate()
            .map(|(traj_idx, pos)| {
                let src_end_idx =
                    ((traj_idx + 1) as f32 * trajectory.δt * sampling_rate).round() as usize - 1;
                (pos, src_end_idx)
            })
            .take_while(|(_, src_end_idx)| *src_end_idx < src.len())
            .chain(std::iter::once((final_position, usize::MAX)));

        let (mut position, mut src_end_idx) = position_with_src_end_idx.next().unwrap();
        let mut hrir = self.get_hrir(position);

        let mut h = |n: usize| {
            if n > src_end_idx {
                (position, src_end_idx) = position_with_src_end_idx.next().unwrap();
                hrir = self.get_hrir(position);
            }
            hrir
        };

        // LTV convolution starts here
        //
        let x = src.cha_uc(0);

        if x.is_empty() || self.hrir_size == 0 {
            return StereoAudioBuf::new_empty();
        }

        let max_n = x.len() + self.hrir_size - 1;

        let mut out = StereoAudioBuf::with_capacity(max_n, src.sampling_rate());

        unsafe {
            let y_left_ptr: *mut f32 = out.channels[0].as_mut_ptr();
            let y_right_ptr: *mut f32 = out.channels[1].as_mut_ptr();

            let mut y_left: f32 = 0.0;
            let mut y_right: f32 = 0.0;

            for n in 0..max_n {
                let h_n = h(n); // time-varying impulse response at "time" n
                let h_n_len = h_n.len();
                let h_n_left = h_n.cha_uc(0);
                let h_n_right = h_n.cha_uc(1);

                // k ∈ [max(0, n - h_n_len + 1), min(n+1, x.len()))
                let k_start = (n + 1).saturating_sub(h_n_len);
                let k_end = usize::min(x.len(), n + 1);

                for k in k_start..k_end {
                    y_left += x.get_unchecked(k) * h_n_left.get_unchecked(n - k);
                    y_right += x.get_unchecked(k) * h_n_right.get_unchecked(n - k);
                }

                y_left_ptr.add(n).write(y_left);
                y_left = 0.0;
                y_right_ptr.add(n).write(y_right);
                y_right = 0.0;
            }

            out.channels[0].set_len(max_n);
            out.channels[1].set_len(max_n);
        }

        out
    }

    /// Accounts for parallax
    pub fn get_hrir<'a>(&'a self, pos: impl Into<Cart3D>) -> AudioBufferSlice<'a, 2> {
        let pos = pos.into();
        if pos.is_nan() {
            println!("Cart3D is nan: {pos:?}")
        }

        #[rustfmt::skip]
        let inner_compute = |ear: Cart3D| {
            if pos.x.classify() == FpCategory::Zero
                && pos.y.classify() == FpCategory::Zero
                && pos.z.classify() == FpCategory::Zero
            {
                return AudioBufferSlice {
                    channels: [&[1.0], &[1.0]],
                    sampling_rate: None,
                };
            }

            let mut slope_xy = ((pos.y - ear.y) / (pos.x - ear.x)).clamp(-1E6, 1E6);
            if slope_xy.is_nan() { slope_xy = 1E6 }
            let ic_y = ear.y - slope_xy * ear.x;


            let mut slope_xz = ((pos.z - ear.z) / (pos.x - ear.x)).clamp(-1E6, 1E6);
            if slope_xy.is_nan() { slope_xz = 0.0 }
            let ic_z = ear.z - slope_xz * ear.x;

            let a = 1.0 + slope_xy.powi(2) + slope_xz.powi(2);
            let b = 2.0 * slope_xy * ic_y + 2.0 * slope_xz * ic_z;
            let c = ic_y.powi(2) + ic_z.powi(2) - self.hrir_radius.powi(2);

            let x1 = (-b + (b.powi(2) - 4.0 * a * c).sqrt()) / (2.0 * a);
            let xyz_1 = Cart3D::new(x1, (slope_xy * x1) + ic_y, (slope_xz * x1) + ic_z);

            let x2 = (-b - (b.powi(2) - 4.0 * a * c).sqrt()) / (2.0 * a);
            let xyz_2 = Cart3D::new(x2, slope_xy * x2 + ic_y, slope_xz * x2 + ic_z);

            let query: Shell2D = if pos.dist(xyz_1) < pos.dist(xyz_2) {
                xyz_1.into()
            } else {
                xyz_2.into()
            };

            if query.is_nan() {
                println!("Shell2D is nan: {query:?}\n{pos:?} -> {xyz_1:?}, {xyz_2:?}")
            }
            self.hrir_tree
                .nearest_neighbor(&query.into())
                .unwrap()
                .data
                .as_ref()
        };

        AudioBufferSlice::from([
            inner_compute(self.left_ear_pos).channels[0],
            inner_compute(self.left_ear_pos).channels[1],
        ])
    }

    /*
    pub fn run_old<T>(&self, src: &MonoAudioBuf, trajectory: Trajectory<T>) -> StereoAudioBuf
    where
        T: Into<Cart3D> + Copy,
    {
        let src_sampling_rate = src.sampling_rate().expect("No sampling rate");
        assert_eq!(self.hrir_sampling_rate, src_sampling_rate);

        let block_size = (src_sampling_rate / 10) as usize;
        let time_size: Seconds = 1.0 / 10.0;
        assert!(time_size > trajectory.δt);

        let to_traj_index =
            |idx: usize| ((time_size / trajectory.δt) * idx as f32).floor() as usize;

        let (blocks, rem) = src.as_blocks_strict(block_size);

        let iter = blocks
            .into_iter()
            .chain(rem.as_ref().map(|buf| buf.as_ref()))
            .enumerate()
            .map(|(i, block)| {
                let hrir = self.get_hrir(*trajectory.path.get(to_traj_index(i)).unwrap());
                block.convolve(&hrir)
            });

        AudioBuffer::<2>::crossfade_concatenate(
            iter,
            (self.hrir_size as f32 * 1.0).floor() as usize,
        )
        .with_sampling_rate(Some(src_sampling_rate))
    }
    */
}
