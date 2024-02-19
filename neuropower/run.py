from argparse import ArgumentParser

import nibabel as nib

from neuropower import neuropowermodels


def get_parser():
    parser = ArgumentParser(description="Power analyses for neuroimaging data")
    parser.add_argument("in_file", action="store", help="Input image.")
    parser.add_argument("--mask_img", action="store", required=False, help="Mask image.")
    parser.add_argument(
        "--n",
        action="store",
        type=int,
        required=False,
        help="Total sample size from analysis.",
    )
    parser.add_argument(
        "--datatype",
        choices=["z", "t"],
        default="t",
        required=False,
        help="Data type of input image.",
    )
    parser.add_argument(
        "--design",
        choices=["one-sample", "two-sample"],
        default="one-sample",
        required=False,
        help="Design of analysis from input image.",
    )
    parser.add_argument(
        "--exc",
        action="store",
        type=float,
        default=0.001,
        required=False,
        help="Z-threshold (excursion threshold).",
    )
    parser.add_argument(
        "--alpha",
        action="store",
        type=float,
        default=0.05,
        required=False,
        help="Desired alpha.",
    )
    parser.add_argument(
        "--method",
        action="store",
        choices=["RFT", "CS"],
        default="RFT",
        required=False,
        help="Multiple comparisons correction method.",
    )
    parser.add_argument(
        "--n_iters",
        action="store",
        type=int,
        default=100,
        required=False,
        help="Number of iterations.",
    )
    parser.add_argument(
        "--seed", action="store", type=int, required=False, help="MRandom seed."
    )
    parser.add_argument(
        "--fwhm",
        action="append",
        type=float,
        nargs="+",
        default=[8, 8, 8],
        required=False,
        help="A list of FWHM values in mm of length 3.",
    )

    return parser


def main():
    opts = get_parser().parse_args()

    # Load data
    img = nib.load(opts.in_file)
    spm = img.get_data()
    mask = (spm != 0).astype(int)
    mask_img = nib.Nifti1Image(mask, img.affine)

    # Run analysis
    params, peak_df, power_df = neuropowermodels.run_power_analysis(
        input_img=img,
        n=opts.n,
        dtype=opts.datatype,
        design=opts.design,
        fwhm=opts.fwhm,
        mask_img=mask_img,
        exc=opts.exc,
        alpha=opts.alpha,
        method=opts.method,
        n_iters=opts.n_iters,
        seed=opts.seed,
    )

    print(power_df)


if __name__ == "__main__":
    main()
