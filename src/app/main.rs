use asearmetry::{
    io::load_hrir,
    signal::{AudioSignalConvolution, AudioSignalCore},
};

fn main() {
    let sig = asearmetry::io::read_audio_file("audio/sample_03.wav").unwrap();
    println!("{}", sig.len());

    let hrir = load_hrir().unwrap();

    let out = sig.convolve(&hrir).into_audio_buf(sig.sample_rate());
    println!("{}", out.len());

    asearmetry::io::write_audio_file("audio/out.wav", &out).unwrap();
}
