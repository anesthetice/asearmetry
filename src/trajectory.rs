use crate::{
    coordinates::{Cart3D, Sphere3D},
    math::Seconds,
};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Trajectory<T> {
    pub path: Vec<T>,
    pub δt: Seconds,
}

impl<T> Trajectory<T> {
    pub fn delta_t(&self) -> Seconds {
        self.δt
    }

    pub fn from_equations(mut eq: impl FnMut(Seconds) -> T, t_tot: Seconds, δt: Seconds) -> Self {
        let path_len = (t_tot / δt).ceil() as usize;
        let mut path = Vec::with_capacity(path_len);

        for i in 0..path_len {
            path.push(eq(δt * i as f64));
        }

        Self { path, δt }
    }

    pub fn with_basis<U: From<T>>(self) -> Trajectory<U> {
        Trajectory {
            path: self.path.into_iter().map(U::from).collect(),
            δt: self.δt,
        }
    }

    pub fn downsample(self, δt_desired: Seconds) -> Self
    where
        T: Copy + core::ops::Add<T, Output = T> + core::ops::Mul<f64, Output = T>,
    {
        assert!(δt_desired >= self.δt);
        let δt_i = self.δt;
        let δt_f = δt_desired;

        let new_path_len = (self.path.len() as f64 * δt_i / δt_f).ceil() as usize;
        let mut new_path = Vec::with_capacity(new_path_len);

        (0..new_path_len).for_each(|idx_f| {
            let idx_i_approx = idx_f as f64 * δt_f / δt_i;

            let idx_i_left = idx_i_approx.floor() as usize;
            let weight_left = 1.0 - f64::abs(idx_i_approx - idx_i_approx.floor());
            let point_left = self.path[idx_i_left] * weight_left;

            let idx_i_right = idx_i_approx.ceil() as usize;
            let weight_right = 1.0 - weight_left;
            let point_right = self.path[idx_i_right] * weight_right;

            new_path.push(point_left + point_right);
        });

        Self {
            path: new_path,
            δt: δt_f,
        }
    }

    #[cfg(feature = "plot")]
    pub fn plot(&self)
    where
        T: Into<Cart3D> + Copy,
    {
        use itertools::Itertools;
        use kuva::prelude::*;

        let (data_xy, data_xz) = self
            .path
            .iter()
            .copied()
            .map(|coord| {
                let xyz = coord.into();
                ((xyz.x, xyz.y), (xyz.x, xyz.z))
            })
            .multiunzip::<(Vec<(f64, f64)>, Vec<(f64, f64)>)>();

        let palette = Palette::wong();

        let compute_annot = |points: &[(f64, f64)]| {
            let (x_i, y_i) = *points.first().unwrap();

            let start_annot = TextAnnotation::new("start", x_i - 0.5, y_i - 0.5)
                .with_arrow(x_i - 0.05, y_i - 0.05)
                .with_color("blue")
                .with_font_size(11);

            let (x_f, y_f) = *points.last().unwrap();

            let end_annot = TextAnnotation::new("end", x_f - 0.5, y_f - 0.5)
                .with_arrow(x_f - 0.05, y_f - 0.05)
                .with_color("red")
                .with_font_size(11);

            (start_annot, end_annot)
        };

        // --- xy part ---
        let xy_annot = compute_annot(&data_xy);

        let xy_plot: Vec<Plot> = vec![
            LinePlot::new()
                .with_data(data_xy)
                .with_color(&palette.colors()[0])
                .into(),
        ];

        let xy_layout = Layout::auto_from_plots(&xy_plot)
            .with_x_label("x")
            .with_y_label("y")
            .with_reference_line(
                ReferenceLine::vertical(0.0)
                    .with_color("black")
                    .with_stroke_width(1.0),
            )
            .with_reference_line(
                ReferenceLine::horizontal(0.0)
                    .with_color("black")
                    .with_stroke_width(1.0),
            )
            .with_axis_line_width(0.0)
            .with_annotation(xy_annot.0)
            .with_annotation(xy_annot.1);

        // --- xz part ---
        let xz_annot = compute_annot(&data_xz);

        let xz_plot: Vec<Plot> = vec![
            LinePlot::new()
                .with_data(data_xz)
                .with_color(&palette.colors()[1])
                .into(),
        ];

        let xz_layout = Layout::auto_from_plots(&xz_plot)
            .with_x_label("x")
            .with_y_label("z")
            .with_reference_line(
                ReferenceLine::vertical(0.0)
                    .with_color("black")
                    .with_stroke_width(1.0),
            )
            .with_reference_line(
                ReferenceLine::horizontal(0.0)
                    .with_color("black")
                    .with_stroke_width(1.0),
            )
            .with_axis_line_width(0.0)
            .with_annotation(xz_annot.0)
            .with_annotation(xz_annot.1);

        let scene = Figure::new(1, 2) // 1 row, 2 columns
            .with_plots(vec![xy_plot, xz_plot])
            .with_layouts(vec![xy_layout, xz_layout])
            .with_labels() // bold A, B panel labels
            .render();

        let svg = SvgBackend.render_scene(&scene);

        std::fs::write("trajectory.svg", svg).unwrap();
    }
}

impl Trajectory<Cart3D> {
    pub fn from_diff_equations(
        eq: impl Fn(Cart3D) -> Cart3D,
        init: Cart3D,
        t_tot: Seconds,
        δt: Seconds,
    ) -> Self {
        let path_len = (t_tot / δt).ceil() as usize;
        let mut path = Vec::with_capacity(path_len);

        let mut position = init;
        path.push(position);

        for _ in 1..(path_len as u64) {
            position += eq(position) * δt;
            path.push(position);
        }

        Self { path, δt }
    }
}

impl Trajectory<Sphere3D> {
    pub fn from_diff_equations(
        eq: impl Fn(Sphere3D) -> Sphere3D,
        init: Sphere3D,
        t_tot: Seconds,
        δt: Seconds,
    ) -> Self {
        let path_len = (t_tot / δt).ceil() as usize;
        let mut path = Vec::with_capacity(path_len);

        let mut position = init;
        path.push(position);

        for _ in 1..(path_len as u32) {
            position += eq(position) * δt;
            // TODO: maybe add constraints here?
            path.push(position);
        }

        Self { path, δt }
    }
}
