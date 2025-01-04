from prelude import pf, np, jax, jnp, jsp, PI
import trajectory as tj
from func import concat_signals, linear
from utils import round_up

def main():
    filepath = "./dtfs/dtf_d_nh1189.sofa"

    hrirs, source_coords, receiver_coords = pf.io.read_sofa(filepath)

    if not isinstance(hrirs, pf.Signal):
       raise ValueError

    inputsig = pf.io.read_audio("./audio/sample_02.opus")
    if inputsig is None:
        raise ValueError

    insig_rate = inputsig.sampling_rate
    insig_duration = inputsig.signal_length
    insig_max = inputsig.n_samples

    print(f"input duration: {insig_duration}")

    inputsig = pf.Signal(
       inputsig.time[0], insig_rate
    )

    #pf.io.write_audio(inputsig, "input.mp3")

    mvmt_freq, mvmt_period = 50, 1/50
    positions = tj.example(mvmt_freq, round_up(insig_duration, decimals=2)).as_coordinates()
    pos_len = positions.cshape[0]
    source_coords_radius = jnp.mean(source_coords.radius)
    batch_size = 20

    partials = []
    mvmt_to_insig = int(mvmt_period * insig_rate)

    for idx in range(0, pos_len, batch_size):
        pos = positions[idx:idx+batch_size]
        pos_len = pos.cshape[0]
        interp_hrirs = linear(positions[idx:idx+batch_size], source_coords, source_coords_radius, hrirs)

        for sidx in range(0, pos_len):
            start = (idx + sidx) * mvmt_to_insig
            end = start + mvmt_to_insig

            full = pf.dsp.convolve(inputsig, interp_hrirs[sidx, :])
            partial = pf.Signal(
                full.time[:, start:end],
                insig_rate,
            )
            partials.append(partial)

    final = concat_signals(partials)
    pf.io.write_audio(final, "output.mp3")

if __name__ == "__main__":
    main()
