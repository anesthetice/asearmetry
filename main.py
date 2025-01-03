from prelude import pf, np, jax, jnp, jsp, PI
import trajectory as tj
from func import concat_signals, linear

def main():
    filepath = "./dtfs/dtf_d_nh1189.sofa"

    hrirs, source_coords, receiver_coords = pf.io.read_sofa(filepath)

    inputsig = pf.io.read_audio("./audio/sample_02.opus")
    if inputsig is None:
        raise ValueError

    insig_rate = inputsig.sampling_rate
    insig_duration = inputsig.signal_length
    insig_max = inputsig.n_samples

    print(insig_duration)

    # quick hack
    """
    print(inputsig.time.shape)
    inputsig = pf.Signal(
        np.append(inputsig.time[0], np.zeros(3 * insig_rate)), insig_rate
    )
    print(inputsig.time.shape)
    """
    inputsig = pf.Signal(
       inputsig.time[0], insig_rate
    )

    frequency = 50
    positions = tj.example(frequency, insig_duration).as_coordinates()

    source_coords_radius = jnp.mean(source_coords.radius)
    batch_size = 20



    # interp_hrirs = linear(positions.as_coordinates(), source_coords, hrirs)

    pf.io.write_audio(inputsig, "sample_02.wav")


if __name__ == "__main__":
    main()
