"""Cabin coverage and head resolution for a ceiling fisheye camera.

Answers one question that costs nothing to ask and a lot to get wrong after
buying hardware: can a Waveshare IMX219-160 mounted on a bus ceiling actually
see the passengers we intend to count?

Everything here is geometry from published sensor and lens numbers. Nothing is
measured. A real FOV test in a real cabin overrides every number this prints.

Lens model
----------
The 160 deg figure for the IMX219-160 is a DIAGONAL fisheye field of view.
Applying the rectilinear formula `width = 2 * H * tan(fov / 2)` to it is wrong
and inflates coverage badly. We use the equidistant fisheye projection

    r = f * theta

which is the standard first-order model for this lens class, and split the
diagonal FoV into horizontal and vertical half-angles by the sensor aspect
ratio in the image plane. The real lens deviates from equidistant; calibrating
each physical module with the OpenCV fisheye model is still required.

Run:
    python development/scripts/camera_geometry.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Verified hardware numbers -------------------------------------------
# Source: Waveshare IMX219-160 product page, https://www.waveshare.com/imx219-160-camera.htm
SENSOR_W_PX = 3280
SENSOR_H_PX = 2464
DIAGONAL_FOV_DEG = 160.0

# --- Assumptions, NOT verified -------------------------------------------
# Every value in this block must be replaced with a tape-measure reading from
# the actual bus model before any of this is treated as a result.
CEILING_HEIGHT_M = 2.30  # interior floor to ceiling, assumption
STANDING_HEAD_M = 1.70  # top of head of a standing adult, assumption
SEATED_HEAD_M = 1.25  # top of head of a seated adult, assumption
CABIN_WIDTH_M = 2.50  # interior width, assumption
CABIN_LENGTH_M = 11.50  # interior length of a 12 m rigid bus, assumption
HEAD_WIDTH_M = 0.20  # for pixels-per-head, assumption

# Detection floor. Below roughly this many pixels across, a head is not
# reliably detectable by any current method, before motion blur or low light
# are taken into account. Treat as a rule of thumb, not a measured threshold.
MIN_HEAD_PX = 12.0


@dataclass(frozen=True)
class Lens:
    """Equidistant fisheye, characterised by its half-angles."""

    half_diag_deg: float
    half_horiz_deg: float
    half_vert_deg: float
    focal_px: float

    @classmethod
    def from_diagonal(cls, diag_fov_deg: float, w_px: int, h_px: int) -> "Lens":
        half_diag = diag_fov_deg / 2.0
        r_diag_px = math.hypot(w_px, h_px) / 2.0
        # Equidistant: image radius is linear in angle, so the half-angle along
        # each axis scales with that axis's share of the image half-diagonal.
        half_horiz = half_diag * (w_px / 2.0) / r_diag_px
        half_vert = half_diag * (h_px / 2.0) / r_diag_px
        focal_px = r_diag_px / math.radians(half_diag)
        return cls(half_diag, half_horiz, half_vert, focal_px)

    def px_per_rad(self) -> float:
        return self.focal_px


def nadir_footprint(lens: Lens, drop_m: float) -> tuple[float, float]:
    """Coverage of a plane `drop_m` below a straight-down camera.

    Returns (length_along_sensor_long_axis, length_along_short_axis) in metres.
    """
    long_side = 2.0 * drop_m * math.tan(math.radians(lens.half_horiz_deg))
    short_side = 2.0 * drop_m * math.tan(math.radians(lens.half_vert_deg))
    return long_side, short_side


def oblique_reach(lens: Lens, drop_m: float, tilt_deg: float) -> tuple[float, float]:
    """Near and far reach along the aisle for a camera tilted off nadir.

    `tilt_deg` is measured from straight down. Returns (near_m, far_m) as
    horizontal distances from the point directly under the camera. A far value
    of `inf` means the field of view passes the horizontal and the geometry
    stops being the limit.
    """
    half = lens.half_horiz_deg
    near_angle = tilt_deg - half
    far_angle = tilt_deg + half
    near = drop_m * math.tan(math.radians(near_angle))
    far = math.inf if far_angle >= 89.5 else drop_m * math.tan(math.radians(far_angle))
    return near, far


def head_px(lens: Lens, distance_m: float) -> float:
    """Apparent head width in pixels at a given slant distance."""
    return lens.px_per_rad() * (HEAD_WIDTH_M / distance_m)


def max_useful_distance(lens: Lens) -> float:
    """Slant distance at which a head falls to MIN_HEAD_PX across."""
    return lens.px_per_rad() * HEAD_WIDTH_M / MIN_HEAD_PX


def slant_distance(drop_m: float, horizontal_m: float) -> float:
    return math.hypot(drop_m, horizontal_m)


def rule(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    lens = Lens.from_diagonal(DIAGONAL_FOV_DEG, SENSOR_W_PX, SENSOR_H_PX)

    rule("Lens, derived from the published diagonal FoV")
    print(f"diagonal FoV          {2 * lens.half_diag_deg:.1f} deg (published)")
    print(f"horizontal FoV        {2 * lens.half_horiz_deg:.1f} deg (derived)")
    print(f"vertical FoV          {2 * lens.half_vert_deg:.1f} deg (derived)")
    print(f"angular resolution    {lens.px_per_rad() * math.pi / 180:.1f} px/deg")
    print()
    print("For contrast, the wrong rectilinear reading of the same 160 deg:")
    wrong = 2.0 * math.tan(math.radians(80.0))
    right = 2.0 * math.tan(math.radians(lens.half_horiz_deg))
    print(
        f"  rectilinear-160 would predict {wrong:.2f} m of coverage per metre of drop"
    )
    print(f"  equidistant fisheye predicts  {right:.2f} m per metre of drop, long axis")
    print(f"  overstatement factor          {wrong / right:.2f}x")

    rule("Straight-down ceiling mount, coverage at each head plane")
    print(f"assumed ceiling height {CEILING_HEIGHT_M:.2f} m")
    print()
    print(f"{'plane':<22}{'drop m':>8}{'along bus m':>14}{'across bus m':>14}")
    for name, head_h in (
        ("floor", 0.0),
        ("seated heads", SEATED_HEAD_M),
        ("standing heads", STANDING_HEAD_M),
    ):
        drop = CEILING_HEIGHT_M - head_h
        along, across = nadir_footprint(lens, drop)
        across_used = min(across, CABIN_WIDTH_M)
        print(f"{name:<22}{drop:>8.2f}{along:>14.2f}{across_used:>14.2f}")
    print()
    print("Long sensor axis is assumed mounted along the bus. Across-bus figures")
    print(f"are clipped at the assumed {CABIN_WIDTH_M:.2f} m cabin width.")

    rule("How many straight-down cameras a whole cabin would need")
    for name, head_h in (
        ("seated heads", SEATED_HEAD_M),
        ("standing heads", STANDING_HEAD_M),
    ):
        drop = CEILING_HEIGHT_M - head_h
        along, across = nadir_footprint(lens, drop)
        n_len = math.ceil(CABIN_LENGTH_M / along)
        n_wide = math.ceil(CABIN_WIDTH_M / across) if across < CABIN_WIDTH_M else 1
        print(
            f"{name:<18} {along:.2f} m x {across:.2f} m per camera "
            f"-> {n_len} along x {n_wide} across = {n_len * n_wide} cameras"
        )
    print()
    print("This ignores overlap needed for stitching, so it is a lower bound.")

    rule("Tilted mount at the front of the cabin, reach along the aisle")
    drop_standing = CEILING_HEIGHT_M - STANDING_HEAD_M
    print(f"camera above the standing head plane by {drop_standing:.2f} m")
    print()
    print(
        f"{'tilt from nadir':>16}{'near edge m':>14}{'far edge m':>14}{'covers cabin':>14}"
    )
    for tilt in (0, 15, 30, 45, 60, 75):
        near, far = oblique_reach(lens, drop_standing, float(tilt))
        far_s = "unbounded" if math.isinf(far) else f"{far:.2f}"
        covers = "yes" if (math.isinf(far) or far >= CABIN_LENGTH_M) else "no"
        print(f"{tilt:>13} deg{near:>14.2f}{far_s:>14}{covers:>14}")
    print()
    print("Negative near edge means the camera also sees behind its own mounting")
    print("point. Unbounded far edge means the field of view crosses the")
    print("horizontal, so geometry stops limiting reach.")

    rule("What actually limits a tilted mount: pixels on head")
    d_max = max_useful_distance(lens)
    print(
        f"head assumed {HEAD_WIDTH_M * 100:.0f} cm across, detection floor {MIN_HEAD_PX:.0f} px"
    )
    print(f"head falls to the floor at a slant distance of {d_max:.1f} m")
    print()
    print(f"{'along aisle m':>15}{'slant m':>10}{'head px':>10}{'usable':>9}")
    for along in (1, 2, 4, 6, 8, 10, 12, 15):
        slant = slant_distance(drop_standing, float(along))
        px = head_px(lens, slant)
        print(
            f"{along:>15}{slant:>10.2f}{px:>10.1f}{'yes' if px >= MIN_HEAD_PX else 'no':>9}"
        )
    print()
    print("Resolution alone does not settle it. At a shallow angle passengers")
    print("occlude each other, and occlusion, not pixels, is what breaks counting")
    print("at the far end of a full bus. Only a real cabin test can measure that.")

    rule("Verdict")
    drop_seated = CEILING_HEIGHT_M - SEATED_HEAD_M
    along_seated, _ = nadir_footprint(lens, drop_seated)
    print(
        f"A single straight-down ceiling camera sees about {along_seated:.1f} m of cabin\n"
        f"length at seated head height, against an assumed {CABIN_LENGTH_M:.1f} m cabin.\n"
        "One nadir camera therefore cannot support a whole-cabin count. The\n"
        "options are a tilted mount, several cameras, or narrowing the product\n"
        "target to a visible zone. This is a geometric argument for planning a\n"
        "FOV test, not a substitute for running one."
    )


if __name__ == "__main__":
    main()
