"""Synthetic checks for CODEC-2's bank-2 paragraph rule."""
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codec2_token_bank import ids_digest, paragraphs  # noqa: E402

BOOK = "\r\n".join([
    "front matter that is not the body",
    "",
    "Chapter I",
    "",
    "THE FIRST TITLE",
    "",
    "",
    "First paragraph, line one,",
    "line two.  Two spaces kept.",
    "",
    "Second paragraph about the Rochester Chapter.",
    "",
    "Chapter II",
    "",
    "THE SECOND TITLE",
    "",
    "Third paragraph.",
    "*** END OF THE PROJECT GUTENBERG EBOOK THE WARDEN ***",
    "licence text that is not the body",
])


class ParagraphRuleTests(unittest.TestCase):
    def test_body_starts_at_chapter_one_and_stops_at_the_end_marker(self):
        paras = paragraphs(BOOK)
        self.assertEqual(paras[0], "First paragraph, line one, line two.  Two spaces kept.")
        self.assertEqual(paras[-1], "Third paragraph.")

    def test_headings_and_their_titles_are_dropped_but_body_mentions_kept(self):
        self.assertEqual(paragraphs(BOOK), [
            "First paragraph, line one, line two.  Two spaces kept.",
            "Second paragraph about the Rochester Chapter.",
            "Third paragraph.",
        ])

    def test_an_edition_without_the_markers_is_refused(self):
        with self.assertRaises(SystemExit):
            paragraphs("no chapters here\n")
        with self.assertRaises(SystemExit):
            paragraphs("Chapter I\n\ntext\n")

    def test_the_digest_is_over_u32_little_endian(self):
        self.assertEqual(ids_digest([2, 258]), hashlib.sha256(b"\x02\x00\x00\x00\x02\x01\x00\x00").hexdigest())


if __name__ == "__main__":
    unittest.main()
