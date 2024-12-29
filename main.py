from numpy import pi
import pyfar as pf

def main():
    filepath = "./dtfs/dtf_d_nh1189.sofa"

    hrirs, source_coords, receiver_coords = pf.io.read_sofa(filepath)

    audio = pf.io.read_audio("./audio/sample_01.opus")
    if audio is None:
        raise ValueError


    front = pf.Coordinates.from_cylindrical(0, 2, 2)
    index, _ = source_coords.find_nearest(front)
    bin_audio = pf.dsp.convolve(audio, hrirs[index])

    left = pf.Coordinates.from_cylindrical(+pi/2, 2, 2)
    index, _ = source_coords.find_nearest(left)
    bin_audio += pf.dsp.convolve(audio, hrirs[index])

    right = pf.Coordinates.from_cylindrical(-pi/2, 2, 2)
    index, _ = source_coords.find_nearest(right)
    bin_audio += pf.dsp.convolve(audio, hrirs[index])

    pf.io.write_audio(bin_audio, "sample_01.wav")


if __name__ == "__main__":
    main()
