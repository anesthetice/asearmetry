use std::f32::consts::PI;

use asearmetry::{
    Seconds,
    brp::Binauralizer,
    coordinates::Sphere3D,
    io,
    signal::{AudioSignalConvolution, AudioSignalCore, ChannelBuffers, StereoAudioBuf},
    trajectory::Trajectory,
};

fn main() -> anyhow::Result<()> {
    let input = io::read_audio_file("audio/dune_sample.wav")?.low_pass(10000.0);

    let hrir_data = io::load_hrir_data("sofa_conversion/output/hrtf.parquet")?;

    let binaur = Binauralizer::new(hrir_data);

    println!(
        "input freq: {}; HRIR freq: {}",
        input.sample_rate(),
        binaur.hrir_sample_rate
    );

    let trajectory = Trajectory::from_equations(
        |t| Sphere3D::new(1.0, t * PI / 10.0, t * PI / 10.0).and_clamp_angles(),
        input.duration(),
        0.05,
    );

    let out_1 = {
        let mut many_buffers: Vec<ChannelBuffers<2>> = Vec::new();

        let mut index: usize = 0;
        let mut time_slice: Seconds = 0.0;
        let mut current = binaur
            .hrir_tree
            .nearest_neighbor(&trajectory.path.first().unwrap().to_shell_point())
            .unwrap();

        for (i, coord) in trajectory.path.iter().enumerate() {
            let new = binaur
                .hrir_tree
                .nearest_neighbor(&coord.to_shell_point())
                .unwrap();

            if new.geom() != current.geom() || i == trajectory.path.len() - 1 {
                println!(
                    "{:?} ; {:?}",
                    trajectory.path[i - 1].to_shell_point(),
                    current.geom()
                );
                let n_samples_to_read = (time_slice * input.sample_rate()).ceil() as usize;
                println!("{}", n_samples_to_read);
                let convolved = input
                    .slice(index..index + n_samples_to_read)
                    .convolve(&current.data);
                many_buffers.push(convolved);

                time_slice = 0.0;
                index += n_samples_to_read;
                current = new;
            }

            time_slice += trajectory.δt;
        }

        <ChannelBuffers<2> as AudioSignalCore<2>>::crossfade_concatenate(
            many_buffers,
            current.data.len() * 1,
        )
        .normalize()
    };

    let trajectory = Trajectory::from_equations(
        |t| Sphere3D::new(1.0, -t * PI / 5.0, (-t * PI / 5.0) + PI / 3.0).and_clamp_angles(),
        input.duration(),
        0.05,
    );

    let out_2 = {
        let mut many_buffers: Vec<ChannelBuffers<2>> = Vec::new();

        let mut index: usize = 0;
        let mut time_slice: Seconds = 0.0;
        let mut current = binaur
            .hrir_tree
            .nearest_neighbor(&trajectory.path.first().unwrap().to_shell_point())
            .unwrap();

        for (i, coord) in trajectory.path.iter().enumerate() {
            let new = binaur
                .hrir_tree
                .nearest_neighbor(&coord.to_shell_point())
                .unwrap();

            if new.geom() != current.geom() || i == trajectory.path.len() - 1 {
                println!(
                    "{:?} ; {:?}",
                    trajectory.path[i - 1].to_shell_point(),
                    current.geom()
                );
                let n_samples_to_read = (time_slice * input.sample_rate()).ceil() as usize;
                println!("{}", n_samples_to_read);
                let convolved = input
                    .slice(index..index + n_samples_to_read)
                    .convolve(&current.data);
                many_buffers.push(convolved);

                time_slice = 0.0;
                index += n_samples_to_read;
                current = new;
            }

            time_slice += trajectory.δt;
        }

        <ChannelBuffers<2> as AudioSignalCore<2>>::crossfade_concatenate(
            many_buffers,
            current.data.len() * 1,
        )
        .normalize()
    };

    /*
    let trajectory = Trajectory::from_equations(
        |t| {
            if t < input.duration() / 2.0 {
                Sphere3D::new(1.0, PI / 2.0, PI / 2.0)
            } else {
                Sphere3D::new(1.0, -PI / 2.0, PI / 2.0)
            }
            .and_adjust_angles()
        },
        input.duration(),
        0.05,
    );
    */

    /*
    let trajectory = Trajectory::from_equations(
        |t| {
            match t as u32 % 4 {
                0 => Sphere3D::new(1.0, PI / 4.0, PI / 2.0),
                1 => Sphere3D::new(1.0, 3.0 * PI / 4.0, PI / 2.0),
                2 => Sphere3D::new(1.0, -3.0 * PI / 4.0, PI / 2.0),
                3 => Sphere3D::new(1.0, -PI / 4.0, PI / 2.0),
                _ => unreachable!(),
            }
            .and_adjust_angles()
        },
        input.duration(),
        0.05,
    );
    */

    asearmetry::io::write_audio_file(
        "audio/out.wav",
        &out_1
            .merge_with(out_2)
            .normalize()
            .into_audio_buf(input.sample_rate())
            .low_pass(4000.0),
    )
    .unwrap();

    Ok(())
}
