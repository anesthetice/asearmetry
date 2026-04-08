// Modules
mod asp; // audio signal processing
mod conv; // convolution
mod dsp; // discrete signal processing
mod utils;

// Exports
pub use asp::AudioSignal;
pub use dsp::DiscreteSignal;
pub(crate) use utils::DiscreteSignalUtils;

// Local exports
use conv::DefinedConvolution;
