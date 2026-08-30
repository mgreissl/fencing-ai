"""Shared utilities for the Fencing AI pipeline."""

from PIL import Image


def extract_thumbnail(path: str) -> Image.Image:
    """Extract the middle frame from a video as a PIL Image."""
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(path, ctx=cpu(0))
        mid_idx = len(vr) // 2
        frame = vr[mid_idx].asnumpy()
        return Image.fromarray(frame)
    except ImportError:
        import av

        container = av.open(path)
        frames = [f for f in container.decode(container.streams.video[0])]
        container.close()
        mid_idx = len(frames) // 2
        frame = frames[mid_idx].to_ndarray(format="rgb24")
        return Image.fromarray(frame)
