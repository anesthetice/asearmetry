use asearmetry::{io::load_hrir, signal::AudioBufferCore};

fn main() {
    let sig = asearmetry::io::read_audio_file("sample_03.wav").unwrap();
    println!("{}", sig.len());

    let hrir = load_hrir().unwrap();

    let out = hrir.convolve(&sig).into_signal(sig.sampling_rate);
    println!("{}", out.len());

    asearmetry::io::write_audio_file("out.wav", &out).unwrap();
}
