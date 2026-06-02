use super::{clamp_azimuth, clamp_zenith};
use crate::coordinates::{Cart3D, Sphere3D};
use crate::math::Radians;
use std::f64::consts::{PI, TAU};

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Shell2D {
    /// The azimuth angle. θ ∈ (-π, +π]
    pub θ: Radians,
    /// The zenith angle. φ ∈ [0, +π]
    /// Note that φ=0 corresponds to the -z direction, while φ=π corresponds to the +z direction.
    pub φ: Radians,
}

impl std::fmt::Debug for Shell2D {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let write_float = |float: f64, f: &mut std::fmt::Formatter<'_>| {
            if float.abs() > 1E-3 { write!(f, "{:.3}", float) }
            else if float.abs() < 1E-6 { write!(f, "0.0") }
            else { write!(f, "{:.1E}", float) }
        };

        let θ_opi = self.θ / PI;
        write!(f, "(θ: π⋅")?;
        write_float(θ_opi, f)?;

        let φ_opi = self.φ / PI;
        write!(f, ", φ: π⋅")?;
        write_float(φ_opi, f)?;
        write!(f, ")")?;

        Ok(())
    }
}

impl std::fmt::Display for Shell2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self, f)
    }
}

impl Shell2D {
    pub fn new(θ: Radians, φ: Radians) -> Self {
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

    pub fn dist(&self, other: Shell2D) -> f64 {
        let θ1 = self.θ;
        let θ2 = other.θ;
        let θ_diff = f64::min((θ1 - θ2).abs(), TAU - (θ1 - θ2).abs());
        debug_assert!(θ_diff < PI + f64::EPSILON);

        let φ1 = self.φ;
        let φ2 = other.φ;
        let φ_diff = φ1 - φ2;
        debug_assert!(φ_diff < PI + f64::EPSILON);

        (θ_diff.powi(2) + φ_diff.powi(2)).sqrt()
    }

    pub fn is_nan(&self) -> bool {
        self.θ.is_nan() || self.φ.is_nan()
    }

    pub fn is_finite(&self) -> bool {
        self.θ.is_finite() && self.φ.is_finite()
    }

    pub fn is_null(&self) -> bool {
        // Under IEEE 754, -0.0 is equal to 0.0
        self.θ == 0.0 && self.φ == 0.0
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

impl From<Shell2D> for [Radians; 2] {
    fn from(value: Shell2D) -> Self {
        [value.θ, value.φ]
    }
}

impl From<[Radians; 2]> for Shell2D {
    fn from(value: [Radians; 2]) -> Self {
        Self {
            θ: value[0],
            φ: value[1],
        }
    }
}

impl From<&[Radians; 2]> for Shell2D {
    fn from(value: &[Radians; 2]) -> Self {
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
    type Envelope = rstar::AABB<[Radians; 2]>;
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
