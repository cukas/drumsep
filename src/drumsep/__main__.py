"""Allow running drumsep as a module: python -m drumsep"""
from drumsep.cli import main
import sys

sys.exit(main())
