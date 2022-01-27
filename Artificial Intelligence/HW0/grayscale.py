from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from matplotlib import pyplot as plt
from matplotlib.image import imread
from pathlib import Path
import numpy as np


def get_matrix(shape: tuple) -> np.ndarray:
    """ Generates a matrix that can be used to compute the grayscale of an image """
    assert len(shape) == 3 and shape[2] == 3, f"Unexpected shape of image: {shape}"
    # (width x 3) Tensor
    _mat = np.ones((shape[0], 3))
    # Broadcast values for [R, G, B]
    _mat = _mat * np.array([0.2989, 0.5870, 0.1140])
    # Reshape (width x 3 x 1) Tensor
    return np.expand_dims(_mat, axis=2)


def write_grayscale(grayscale: np.ndarray, dest: Path) -> None:
    """ Writes the grayscale image to the destination path """
    grayscale = grayscale / 255
    plt.imsave(str(dest.absolute()), grayscale)
    print(f"Grayscale Image Written To: {dest.absolute()}")


def main(target: Path, dest: Path) -> None:
    image_rbg = imread(str(target.absolute()))
    matrix = get_matrix(image_rbg.shape)
    # (width, height, 3) x (width, 3, 1) -> (width, height, 1)
    grayscale = np.matmul(image_rbg, matrix)
    # (width, height, 1) -> (widght, height, 3) [last dim is repeated]
    grayscale = np.repeat(grayscale, 3, axis=2)
    write_grayscale(grayscale, dest)


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="This program computes the grayscale of a given image and writes the image to a given path"
    )
    parser.add_argument('-p', '--path', type=Path, required=True, dest='target', help=(
        "The path to the image (e.g. /Users/name/Downloads/image.jpeg")
    )
    parser.add_argument('-o', '--outpath', type=Path, required=False, dest='dest', default='outfile.jpeg', help=(
        "Where you would like the grayscale image to be written "
    ))
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    assert args.target.exists(), f"{args.target} does not exist!"
    assert args.target.suffix == args.dest.suffix, f"'{args.target.suffix}' != '{args.dest.suffix}' - Please specify the same target and destination file type"
    main(args.target, args.dest)
