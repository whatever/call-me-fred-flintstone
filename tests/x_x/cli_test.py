import unittest

from x_x.cli import (
    main,
    splash,
)


class TestCLI(unittest.TestCase):
    """Test that the CLI tool still works"""

    def test_splash(self):
        """Test whether the splash screen returns"""

        self.assertTrue(splash("sword"))

    def test_main(self):
        """Test whether the main function runs"""

        main()
