#!/usr/bin/env python3
"""
Run all tests for Gamma-Pareto distribution package.
"""
import pytest
import sys

if __name__ == "__main__":
    # Run tests and exit with appropriate code
    exit_code = pytest.main([
        "test_gamma_pareto",
        "--verbose",
        "--tb=short",
        "--durations=10",
        "-v"
    ])
    sys.exit(exit_code)