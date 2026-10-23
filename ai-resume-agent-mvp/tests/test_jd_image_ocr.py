import asyncio
from io import BytesIO
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi import UploadFile
from PIL import Image

from app.main import parse_jd


def png_bytes():
    stream = BytesIO()
    Image.new("RGB", (40, 30), "white").save(stream, format="PNG")
    return stream.getvalue()


class JobImageParsingTests(unittest.TestCase):
    def test_png_is_sent_to_vision_model_without_tesseract(self):
        upload = UploadFile(filename="job.png", file=BytesIO(png_bytes()))
        with patch(
            "app.main.extract_job_text_from_image",
            new_callable=AsyncMock,
            return_value="招聘岗位：产品经理。负责产品规划和跨团队协作。" * 2,
        ) as vision:
            result = asyncio.run(parse_jd(upload))

        self.assertTrue(result["ok"])
        self.assertEqual(result["filename"], "job.png")
        self.assertIn("产品经理", result["jd_text"])
        vision.assert_awaited_once()
        self.assertEqual(vision.await_args.args[1], "image/png")

    def test_invalid_image_returns_json_api_error(self):
        upload = UploadFile(filename="job.png", file=BytesIO(b"not an image"))
        with patch("app.main.extract_job_text_from_image", new_callable=AsyncMock) as vision:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(parse_jd(upload))

        self.assertEqual(raised.exception.status_code, 422)
        vision.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
