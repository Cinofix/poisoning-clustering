import sys
from optparse import OptionParser

# parse commandline arguments
op = OptionParser()
op.add_option(
    "--device",
    type=str,
    default="cpu",
    help="Set device name on which experiments are performed.",
)
op.add_option(
    "--path",
    type=str,
    default="./experiments/results/",
    help="Destination path, where results will be stored.",
)
op.add_option(
    "--phi",
    type=str,
    default="AMI",
    help="Similarity measure between clustering ['ARI', 'AMI', 'frobenius'].",
)
# op.print_help()
(opts, args) = op.parse_args(sys.argv[1:])
