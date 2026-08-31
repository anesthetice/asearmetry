use num_traits::Zero;
use std::f64::consts::{PI, TAU};
use tap::Pipe;

use crate::{
    coordinates::{Cart3D, Sphere3D},
    math::{Meters, Radians},
};

pub(crate) fn write_float(
    mut float: f64,
    f: &mut std::fmt::Formatter<'_>,
    is_angle: bool,
) -> std::fmt::Result {
    match (is_angle, f.alternate()) {
        (true, true) => {
            float /= PI;
            if float.abs() > 1E-3 {
                write!(f, "{:.3}⋅π", float)
            } else if float.is_zero() {
                write!(f, "0.0")
            } else if float.abs() < 1E-9 {
                write!(f, "~0.0")
            } else {
                write!(f, "{:.1E}⋅π", float)
            }
        }
        (true, false) => {
            float = 360.0 * float / TAU;
            let precision = f.precision().unwrap_or(0);
            match float.signum() {
                1.0 => write!(f, "+{1:.*}°", precision, float),
                -1.0 => write!(f, "{1:.*}°", precision, float),
                _ => write!(f, "NaN"),
            }
        }
        (false, _) => {
            if float.abs() > 1E-3 {
                write!(f, "{:.3}", float)
            } else if float.is_zero() {
                write!(f, "0.0")
            } else if float.abs() < 1E-9 {
                write!(f, "~0.0")
            } else {
                write!(f, "{:.1E}", float)
            }
        }
    }
}

/// Distance with line of sight between a point and another on the surface
/// of a sphere, where the path taken cannot cross through the sphere.
pub fn distance_los_point_to_sphere_surface_point(
    point: impl Into<Cart3D>,
    surface_point: impl Into<Sphere3D>,
) -> Meters {
    let (p, dist_center_to_p, s, dist_center_to_s) = {
        let p_cart = point.into();
        let dist_center_to_p = p_cart.norm();

        let s_sphere = surface_point.into();
        let s_cart = Cart3D::from(s_sphere);
        let dist_center_to_s = s_sphere.r;

        (p_cart, dist_center_to_p, s_cart, dist_center_to_s)
    };

    if dist_center_to_p < dist_center_to_s {
        log::warn!(
            "Cannot compute the distance between the point p = {p}, and the point on the surface of a sphere s = {s}, as the norm of p is smaller than s, defaulting to 0.0"
        );
        return 0.0;
    }

    // The angle 'α' is equal to ∠(OS⃗ ⃗, OP ⃗), where 'O' is the center of the sphere
    // with the position of (x, y, z) = (0, 0, 0) obviously in cartesian coordinates.
    let α: Radians = s.angle_with(p).abs();

    // For any point on the surface of the sphere, the angle 'α_crit' corresponds to the
    // maximum 'α' angle possible between these two points such that the line that connects
    // them does not pass through the sphere. You can think of it as the maximum angle
    // possible given their lengths relative to each other where line-of-sight remains possible.
    let α_crit: Radians = f64::acos(dist_center_to_s / dist_center_to_p);

    if α <= α_crit {
        // We already have line-of-sight, so rather trivial
        p.dist(s)
    } else {
        // The path taken from p to s will "touch" the sphere tangentially
        // as far away from p as possible, therefore α will be reduced by α_crit.

        // Pythagora's theorem (advanced stuff I know) to find the "straight" distance traveled
        let dist_straight = f64::sqrt(dist_center_to_p.powi(2) - dist_center_to_s.powi(2));
        // We multiply the remaining angle by the radius to get the arc length.
        let dist_circular = (α - α_crit) * dist_center_to_s;

        dist_straight + dist_circular
    }
}
