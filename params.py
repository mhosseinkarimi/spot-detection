import argparse

def parse_params():
    parser = argparse.ArgumentParser(
        prog="main",
        description="Detection main program parameters",
        fromfile_prefix_chars="@"
    )

    parser.add_argument("test_root", help="Root directory to all test images.", type=str)
    parser.add_argument(
        "detector", 
        help="The choise of the detection algorithm.", 
        choices=["watershed", "thresholding"],
        type=str
        )
    parser.add_argument(
        "--thresh_type",
        help="Type of thresholding", 
        choices=["local", "global", "global_adaptive", "local_adaptive"],
        default="global",
        type=str
        )
    parser.add_argument(
        "--output_dir", 
        help="Path to the directory that the output is stored", 
        default="/mnt/c/Users/mhuss/OneDrive/Desktop/output",
        type=str
        )
    parser.add_argument(
        "--thresh_steps",
        help="Steps of thresholding windows in x and y directions.",
        nargs="*",
        default=[100, 100],
        type=int
    )
    args = parser.parse_args()
    return args
