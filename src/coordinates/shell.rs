use super::{clamp_azimuth, clamp_zenith};
use crate::coordinates::{Cart3D, Sphere3D};
use std::f32::{
    self,
    consts::{PI, TAU},
};

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Shell2D {
    /// The azimuth angle. θ ∈ (-π, +π]
    pub θ: f32,
    /// The zenith angle. φ ∈ [0, +π]
    /// Note that φ=0 corresponds to the -z direction, while φ=π corresponds to the +z direction.
    pub φ: f32,
}

impl std::fmt::Debug for Shell2D {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let θ_opi = self.θ / PI;
        if θ_opi.abs() == 0.0 { write!(f, "(θ: 0.0")? }
        else if θ_opi.abs() > 0.001 { write!(f, "(θ: π⋅{θ_opi:.3}")? }
        else { write!(f, "(θ: π⋅{θ_opi:.1E}")? };

        let φ_opi = self.φ / PI;
        if φ_opi.abs() == 0.0 { write!(f, ", φ: 0.0)")? }
        if φ_opi.abs() > 0.001 { write!(f, ", φ: π⋅{φ_opi:.3})")? }
        else { write!(f, ", φ: π⋅{φ_opi:.1E})")? };
        Ok(())
    }
}

impl std::fmt::Display for Shell2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self, f)
    }
}

impl Shell2D {
    pub fn new(θ: f32, φ: f32) -> Self {
        Self { θ, φ }
    }

    pub fn clamp_angles_in_place(&mut self) {
        self.θ = clamp_azimuth(self.θ);
        self.φ = clamp_zenith(self.φ)
    }

    pub fn clamp_angles(mut self) -> Self {
        self.clamp_angles_in_place();
        self
    }

    pub fn dist(&self, other: Shell2D) -> f32 {
        let θ1 = self.θ;
        let θ2 = other.θ;
        let θ_diff = f32::min((θ1 - θ2).abs(), TAU - (θ1 - θ2).abs());
        debug_assert!(θ_diff < PI + f32::EPSILON);

        let φ1 = self.φ;
        let φ2 = other.φ;
        let φ_diff = φ1 - φ2;
        debug_assert!(φ_diff < PI + f32::EPSILON);

        (θ_diff.powi(2) + φ_diff.powi(2)).sqrt()
    }

    pub fn is_nan(&self) -> bool {
        self.θ.is_nan() || self.φ.is_nan()
    }
}

impl Default for Shell2D {
    fn default() -> Self {
        Self {
            θ: 0.0,
            φ: PI / 2.0,
        }
    }
}

impl From<Shell2D> for [f32; 2] {
    fn from(value: Shell2D) -> Self {
        [value.θ, value.φ]
    }
}

impl From<[f32; 2]> for Shell2D {
    fn from(value: [f32; 2]) -> Self {
        Self {
            θ: value[0],
            φ: value[1],
        }
    }
}

impl From<&[f32; 2]> for Shell2D {
    fn from(value: &[f32; 2]) -> Self {
        Self {
            θ: value[0],
            φ: value[1],
        }
    }
}

impl From<Sphere3D> for Shell2D {
    fn from(v: Sphere3D) -> Self {
        Self { θ: v.θ, φ: v.φ }
    }
}

impl From<Cart3D> for Shell2D {
    fn from(v: Cart3D) -> Self {
        let r = (v.x.powi(2) + v.y.powi(2) + v.z.powi(2)).sqrt();
        Self {
            θ: v.y.atan2(v.x),
            φ: PI - (v.z / r).acos(),
        }
    }
}

impl rstar::RTreeObject for Shell2D {
    type Envelope = rstar::AABB<[f32; 2]>;
    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_point([self.θ, self.φ])
    }
}

impl rstar::PointDistance for Shell2D {
    fn distance_2(
        &self,
        point: &<Self::Envelope as rstar::Envelope>::Point,
    ) -> <<Self::Envelope as rstar::Envelope>::Point as rstar::Point>::Scalar {
        self.dist(point.into())
    }
}
