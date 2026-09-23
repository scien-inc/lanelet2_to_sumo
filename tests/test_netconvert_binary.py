from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ll2sumo.convert import _resolve_netconvert_binary


class ResolveNetconvertBinaryTest(unittest.TestCase):
    def test_explicit_binary_skips_the_sumo_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            explicit = Path(tmp) / "netconvert"
            explicit.write_text("")
            explicit.chmod(0o755)

            with patch("ll2sumo.convert.checkBinary") as check_binary:
                resolved = _resolve_netconvert_binary(str(explicit))

        self.assertEqual(resolved, str(explicit))
        check_binary.assert_not_called()

    def test_defaults_to_the_sumo_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            found = Path(tmp) / "netconvert"
            found.write_text("")
            found.chmod(0o755)

            with patch("ll2sumo.convert.checkBinary", return_value=str(found)):
                self.assertEqual(_resolve_netconvert_binary(None), str(found))

    def test_resolves_a_bare_name_against_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            on_path = Path(tmp) / "netconvert"
            on_path.write_text("")
            on_path.chmod(0o755)

            # checkBinary falls back to the bare name when it finds nothing.
            with patch("ll2sumo.convert.checkBinary", return_value="netconvert"):
                with patch.dict(os.environ, {"PATH": tmp}):
                    self.assertEqual(_resolve_netconvert_binary(None), str(on_path))

    def test_keeps_an_unresolvable_name_for_the_subprocess_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("ll2sumo.convert.checkBinary", return_value="netconvert"):
                with patch.dict(os.environ, {"PATH": tmp}):
                    self.assertEqual(_resolve_netconvert_binary(None), "netconvert")


if __name__ == "__main__":
    unittest.main()
