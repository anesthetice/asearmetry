/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Modules
mod store;

#[cfg(feature = "polars")]
mod precursor;

// Exports
#[cfg(feature = "polars")]
pub use precursor::BinauralizerPrecursor;

// Imports
use crate::{
    audio::{ASP, AudioBuffer, AudioBufferSlice, StereoAudioBuf},
    coordinates::{Cart3D, Shell2D, Sphere3D},
    math::{Hertz, Meters},
    signal::{_convolve_ltv, LtvFilter},
    trajectory::Trajectory,
};
use itertools::Itertools;
use rstar::{RTree, primitives::GeomWithData};

pub type HrirProjection = GeomWithData<Shell2D, AudioBuffer<2>>;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct Binauralizer {
    pub hrir_rtree: RTree<HrirProjection>,
    pub hrir_radius: Meters,
    pub hrir_size: usize,
    pub hrir_sampling_rate: Hertz,
    pub left_ear_pos: Cart3D,
    pub right_ear_pos: Cart3D,
}

impl Binauralizer {
    pub fn run<A, T>(&self, src: A, trajectory: Trajectory<T>) -> StereoAudioBuf
    where
        A: ASP<1>,
        T: Into<Cart3D> + Copy,
    {
        let sampling_rate = src.sr_or_panic();
        approx::assert_abs_diff_eq!(self.hrir_sampling_rate, sampling_rate, epsilon = 0.1);

        let src_final_pos = *trajectory.path.last().unwrap();
        let src_pos_with_last_idx_vec = trajectory
            .path
            .into_iter()
            .enumerate()
            .map(|(traj_idx, pos)| {
                let chunk_last_idx =
                    ((traj_idx + 1) as f64 * trajectory.δt * sampling_rate).round() as usize - 1;
                (pos, chunk_last_idx)
            })
            .take_while(|(_, chunk_last_idx)| *chunk_last_idx < src.len())
            .chain(std::iter::once((src_final_pos, usize::MAX)))
            .collect_vec();
        assert!(!src_pos_with_last_idx_vec.is_empty());

        let mut out = src.into_owned().into_stereo();

        // --- Adding directionality to sound ---
        //
        println!("HRIR step");
        out = out
            .map_cha_enumerate(|c, cha| {
                let hrir_filter = LtvFilter::new_with_state((
                    src_pos_with_last_idx_vec.clone().into_iter(),
                    0_usize,
                ))
                .update_fn(|n, (src_pos_with_last_idx_iter, chunk_last_idx), hrir| {
                    if n > *chunk_last_idx || *chunk_last_idx == 0 {
                        let (src_pos, new_chunk_last_idx) =
                            src_pos_with_last_idx_iter.next().unwrap();
                        *chunk_last_idx = new_chunk_last_idx;
                        let _ = hrir.replace(self.get_hrir(src_pos, c));
                    }
                })
                .get_fn(|hrir| hrir.cha_uc(0))
                .build();

                _convolve_ltv(cha, hrir_filter)
            })
            .into();
        out.set_sr(sampling_rate);

        // --- Adding back ITD to sound ---
        //
        println!("ITD step");
        const SOUND_VELOCITY_IN_AIR: f64 = 343.0; // meters per second
        out = out
            .map_cha_enumerate(|c, cha| {
                let rcv_pos = self.get_ear_pos(c);

                let delay_filter = LtvFilter::new_with_state((
                    src_pos_with_last_idx_vec.clone().into_iter(),
                    0_usize,
                ))
                .update_fn(|n, (src_pos_with_last_idx_iter, chunk_last_idx), filter| {
                    if n > *chunk_last_idx || *chunk_last_idx == 0 {
                        let (src_pos, new_chunk_last_idx) =
                            src_pos_with_last_idx_iter.next().unwrap();
                        *chunk_last_idx = new_chunk_last_idx;

                        let new_filter = {
                            let dist_to_ear_in_samples = (sampling_rate
                                * (src_pos.into().dist(rcv_pos) / SOUND_VELOCITY_IN_AIR))
                                .round()
                                as usize;
                            let mut temp = vec![0.0_f32; dist_to_ear_in_samples.saturating_sub(1)];
                            temp.push(1.0);
                            temp
                        };

                        let _ = filter.replace(new_filter);
                    }
                })
                .get_fn(|filter| filter.as_slice())
                .build();

                _convolve_ltv(cha, delay_filter)
            })
            .into();
        out.set_sr(sampling_rate);

        out
    }

    /// Returns the coordinates for the point on the shell closest to the source,
    /// and that intersects with the infinite line that passes through both the source
    /// and receiver.
    /// In this context, `strl` stands for source-to-receiver line.
    pub fn project_strl_onto_shell(
        &self,
        src_pos: impl Into<Cart3D>,
        rcv_pos: impl Into<Cart3D>,
    ) -> Shell2D {
        let src_pos = src_pos.into();
        let rcv_pos = rcv_pos.into();

        #[cfg(debug_assertions)]
        if src_pos.is_nan() {
            eprintln!("Warning, the position of the source is NaN: {src_pos}");
        } else if !src_pos.is_finite() {
            eprintln!("Warning, the position of the source is not finite: {src_pos}");
        } else if src_pos.is_null() {
            eprintln!("Warning, the position of the source is null");
        }

        debug_assert!(
            Sphere3D::from(rcv_pos).r <= self.hrir_radius,
            "The receiver cannot be outside the HRIR shell"
        );

        let mut slope_xy = ((src_pos.y - rcv_pos.y) / (src_pos.x - rcv_pos.x)).clamp(-1E6, 1E6);
        if slope_xy.is_nan() {
            slope_xy = 1E6
        }
        let ic_y = rcv_pos.y - slope_xy * rcv_pos.x;

        let mut slope_xz = ((src_pos.z - rcv_pos.z) / (src_pos.x - rcv_pos.x)).clamp(-1E6, 1E6);
        if slope_xy.is_nan() {
            slope_xz = 0.0
        }
        let ic_z = rcv_pos.z - slope_xz * rcv_pos.x;

        let a = 1.0 + slope_xy.powi(2) + slope_xz.powi(2);
        let b = 2.0 * slope_xy * ic_y + 2.0 * slope_xz * ic_z;
        let c = ic_y.powi(2) + ic_z.powi(2) - self.hrir_radius.powi(2);

        let x1 = (-b + (b.powi(2) - 4.0 * a * c).sqrt()) / (2.0 * a);
        let xyz_1 = Cart3D::new(x1, (slope_xy * x1) + ic_y, (slope_xz * x1) + ic_z);

        #[cfg(debug_assertions)]
        if !xyz_1.is_finite() {
            eprintln!("Warning, the shell coordinate (`xyz_1`) is not finite: {xyz_1}");
        }

        let x2 = (-b - (b.powi(2) - 4.0 * a * c).sqrt()) / (2.0 * a);
        let xyz_2 = Cart3D::new(x2, slope_xy * x2 + ic_y, slope_xz * x2 + ic_z);

        #[cfg(debug_assertions)]
        if !xyz_2.is_finite() {
            eprintln!("Warning, the shell coordinate (`xyz_2`) is not finite: {xyz_2}");
        }

        if src_pos.dist(xyz_1) < src_pos.dist(xyz_2) {
            xyz_1.into()
        } else {
            xyz_2.into()
        }
    }

    /// Get the ear position by channel index. Passing c=0 (c=1) will return
    /// the position of the left (right) ear; function will panic otherwise.
    pub fn get_ear_pos(&self, c: usize) -> Cart3D {
        match c {
            0 => self.left_ear_pos,
            1 => self.right_ear_pos,
            other => panic!(
                "The channel with index {other} does not correspond to either the left (0) or right (1) channels and thus does not have an associated ear position."
            ),
        }
    }

    /// Get the HRIR that best corresponds the situation where a sound emitted
    /// at `src_pos` is heard by the ear characterized by its channel index `c`
    /// (0 for left, 1 for right). Accounts for parallax.
    pub fn get_hrir<'a>(&'a self, src_pos: impl Into<Cart3D>, c: usize) -> AudioBufferSlice<'a, 1> {
        let rcv_pos = self.get_ear_pos(c);

        let query = self.project_strl_onto_shell(src_pos, rcv_pos);

        #[cfg(debug_assertions)]
        if !query.is_finite() {
            eprintln!("Warning, the query is not finite: {query}");
        }

        self.hrir_rtree
            .nearest_neighbor(query.into())
            .unwrap()
            .data
            .view()
            .index_cha(c)
    }

    /// Self-explanatory, accounts for parallax, see documentation for [`Self::get_hrir`].
    pub fn get_hrir_left<'a>(&'a self, src_pos: impl Into<Cart3D>) -> AudioBufferSlice<'a, 1> {
        self.get_hrir(src_pos, 0)
    }

    /// Self-explanatory, accounts for parallax, see documentation for [`Self::get_hrir`].
    pub fn get_hrir_right<'a>(&'a self, src_pos: impl Into<Cart3D>) -> AudioBufferSlice<'a, 1> {
        self.get_hrir(src_pos, 1)
    }

    /// Self-explanatory, accounts for parallax, see documentation for [`Self::get_hrir`].
    pub fn get_hrir_both<'a>(
        &'a self,
        src_pos: impl Into<Cart3D> + Copy,
    ) -> AudioBufferSlice<'a, 2> {
        AudioBufferSlice::<'a, 2>::stack_ref(
            self.get_hrir_left(src_pos),
            self.get_hrir_right(src_pos),
        )
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
