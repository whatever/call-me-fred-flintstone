import unittest


from x_x import x_x


class TestX_X(unittest.TestCase):
    """Test x_x"""

    def test_x_x_structure(self):
        """Test whether x_x() is structured correctly."""

        for i in range(1000):
            smiley = x_x()
            self.assertEqual(len(smiley), 3)
            self.assertEqual(smiley[1], "_")
            self.assertIn(smiley[0], "xXOo0Uu@*^-")
            self.assertIn(smiley[2], "xXOo0Uu@*^-")


if __name__ == "__main__":
    unittest.main()
