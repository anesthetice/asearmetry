// Modules
mod cartesian;
mod shell;
mod spherical;

// Exports
pub use cartesian::Cart3D;
pub use shell::Shell2D;
pub use spherical::{Sphere3D, clamp_azimuth, clamp_zenith};

#[cfg(test)]
mod test {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    #[rustfmt::skip]
    fn cart_to_sphere_and_back() {
        let q1_above_cart = Cart3D::new(2.0, 2.0, 1.0);
        let q2_above_cart = Cart3D::new(-2.0, 2.0, 1.0);
        let q3_above_cart = Cart3D::new(-2.0, -2.0, 1.0);
        let q4_above_cart = Cart3D::new(2.0, -2.0, 1.0);

        let zenith = (PI / 2.0) + f32::asin(1.0 / 3.0);

        let q1_above_sphere = Sphere3D::new(3.0, PI / 4.0, zenith);
        let q2_above_sphere = Sphere3D::new(3.0, 3.0 * PI / 4.0, zenith);
        let q3_above_sphere = Sphere3D::new(3.0, -3.0 * PI / 4.0, zenith);
        let q4_above_sphere = Sphere3D::new(3.0, -PI / 4.0, zenith);

        approx::assert_abs_diff_eq!(Sphere3D::from(q1_above_cart), q1_above_sphere, epsilon=1e-6);
        approx::assert_abs_diff_eq!(Sphere3D::from(q2_above_cart), q2_above_sphere, epsilon=1e-6);
        approx::assert_abs_diff_eq!(Sphere3D::from(q3_above_cart), q3_above_sphere, epsilon=1e-6);
        approx::assert_abs_diff_eq!(Sphere3D::from(q4_above_cart), q4_above_sphere, epsilon=1e-6);

        approx::assert_abs_diff_eq!(q1_above_cart, Cart3D::from(q1_above_sphere), epsilon=1e-6);
        approx::assert_abs_diff_eq!(q2_above_cart, Cart3D::from(q2_above_sphere), epsilon=1e-6);
        approx::assert_abs_diff_eq!(q3_above_cart, Cart3D::from(q3_above_sphere), epsilon=1e-6);
        approx::assert_abs_diff_eq!(q4_above_cart, Cart3D::from(q4_above_sphere), epsilon=1e-6);

        approx::assert_abs_diff_eq!(Cart3D::from(Sphere3D::from(q1_above_cart)), q1_above_cart, epsilon=1e-6);
        approx::assert_abs_diff_eq!(Cart3D::from(Sphere3D::from(q2_above_cart)), q2_above_cart, epsilon=1e-6);
        approx::assert_abs_diff_eq!(Cart3D::from(Sphere3D::from(q3_above_cart)), q3_above_cart, epsilon=1e-6);
        approx::assert_abs_diff_eq!(Cart3D::from(Sphere3D::from(q4_above_cart)), q4_above_cart, epsilon=1e-6);


        let q1_below_cart = Cart3D::new(2.0, 2.0, -1.0);
        let q2_below_cart = Cart3D::new(-2.0, 2.0, -1.0);
        let q3_below_cart = Cart3D::new(-2.0, -2.0, -1.0);
        let q4_below_cart = Cart3D::new(2.0, -2.0, -1.0);

        let zenith = (PI / 2.0) - f32::asin(1.0 / 3.0);

        let q1_below_sphere = Sphere3D::new(3.0, PI / 4.0, zenith);
        let q2_below_sphere = Sphere3D::new(3.0, 3.0 * PI / 4.0, zenith);
        let q3_below_sphere = Sphere3D::new(3.0, -3.0 * PI / 4.0, zenith);
        let q4_below_sphere = Sphere3D::new(3.0, -PI / 4.0, zenith);

        approx::assert_abs_diff_eq!(Sphere3D::from(q1_below_cart), q1_below_sphere, epsilon=1e-6);
        approx::assert_abs_diff_eq!(Sphere3D::from(q2_below_cart), q2_below_sphere, epsilon=1e-6);
        approx::assert_abs_diff_eq!(Sphere3D::from(q3_below_cart), q3_below_sphere, epsilon=1e-6);
        approx::assert_abs_diff_eq!(Sphere3D::from(q4_below_cart), q4_below_sphere, epsilon=1e-6);

        approx::assert_abs_diff_eq!(q1_below_cart, Cart3D::from(q1_below_sphere), epsilon=1e-6);
        approx::assert_abs_diff_eq!(q2_below_cart, Cart3D::from(q2_below_sphere), epsilon=1e-6);
        approx::assert_abs_diff_eq!(q3_below_cart, Cart3D::from(q3_below_sphere), epsilon=1e-6);
        approx::assert_abs_diff_eq!(q4_below_cart, Cart3D::from(q4_below_sphere), epsilon=1e-6);

        approx::assert_abs_diff_eq!(Cart3D::from(Sphere3D::from(q1_below_cart)), q1_below_cart, epsilon=1e-6);
        approx::assert_abs_diff_eq!(Cart3D::from(Sphere3D::from(q2_below_cart)), q2_below_cart, epsilon=1e-6);
        approx::assert_abs_diff_eq!(Cart3D::from(Sphere3D::from(q3_below_cart)), q3_below_cart, epsilon=1e-6);
        approx::assert_abs_diff_eq!(Cart3D::from(Sphere3D::from(q4_below_cart)), q4_below_cart, epsilon=1e-6);
    }

    #[test]
    fn cart_to_sphere_edge_cases() {
        let x_positive_rest_zero = Cart3D::new(1.0, 0.0, 0.0);
        approx::assert_abs_diff_eq!(
            Sphere3D::from(x_positive_rest_zero),
            Sphere3D::new(1.0, 0.0, PI / 2.0)
        );
        let x_negative_rest_zero = Cart3D::new(-1.0, 0.0, 0.0);
        approx::assert_abs_diff_eq!(
            Sphere3D::from(x_negative_rest_zero),
            Sphere3D::new(1.0, PI, PI / 2.0)
        );

        let y_positive_rest_zero = Cart3D::new(0.0, 1.0, 0.0);
        approx::assert_abs_diff_eq!(
            Sphere3D::from(y_positive_rest_zero),
            Sphere3D::new(1.0, PI / 2.0, PI / 2.0)
        );
        let y_negative_rest_zero = Cart3D::new(0.0, -1.0, 0.0);
        approx::assert_abs_diff_eq!(
            Sphere3D::from(y_negative_rest_zero),
            Sphere3D::new(1.0, -PI / 2.0, PI / 2.0)
        );

        let z_positive_rest_zero = Cart3D::new(0.0, 0.0, 1.0);
        approx::assert_abs_diff_eq!(
            Sphere3D::from(z_positive_rest_zero),
            Sphere3D::new(1.0, 0.0, PI)
        );
        let z_negative_rest_zero = Cart3D::new(0.0, 0.0, -1.0);
        approx::assert_abs_diff_eq!(
            Sphere3D::from(z_negative_rest_zero),
            Sphere3D::new(1.0, 0.0, 0.0)
        );
    }
}
