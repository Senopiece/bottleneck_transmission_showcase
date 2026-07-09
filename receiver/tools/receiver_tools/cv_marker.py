from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from receiver_tools.geometry import (
    LED_FRACTIONS,
    LED_RADIUS_FRACTION,
    SQUARE_SIZE_FRACTION,
    TRIANGLE_SIZE_FRACTION,
    Pose,
    led_scores_from_pose,
    pose_from_markers,
    warp_marker_patch,
)
from receiver_tools.tracker_backend import TrackerResult


ONNX_PATCH_WIDTH = 96
ONNX_PATCH_HEIGHT = 36
REFINE_PATCH_SIZE = (128, 48)


@dataclass
class Candidate:
    pose: Pose
    score: float
    square_score: float
    triangle_score: float
    led_score: float
    geometry_score: float
    template_score: float
    layout_score: float
    strip_score: float
    prior_score: float
    nn_score: float


@dataclass
class RefinedPose:
    pose: Pose
    score: float
    template_score: float
    layout_score: float
    strip_score: float
    border_score: float
    led_center_score: float
    square_anchor_score: float
    triangle_anchor_score: float


class CvMarkerDetector:
    def __init__(self, tracker_model: Path | None = None):
        if tracker_model is None:
            tracker_model = default_tracker_model_path()
        self.onnx_model = None
        if tracker_model and tracker_model.exists():
            if tracker_model.suffix.lower() == ".onnx":
                self.onnx_model = OnnxPatchScorer(tracker_model)

    def detect(self, frame_bgr: np.ndarray, prior_pose: Pose | None = None, tracking: bool = False) -> Candidate | None:
        squares, triangles = find_marker_shapes(frame_bgr)
        best: Candidate | None = None
        cv_candidates: list[Candidate] = []
        h, w = frame_bgr.shape[:2]
        has_patch_model = self.onnx_model is not None
        for sq in squares:
            for tri in triangles:
                pose = pose_from_markers(sq["center"], tri["center"])
                if not (32.0 <= pose.scale_px <= min(w, h) * 0.9):
                    continue
                if sq["size"] <= 0 or tri["size"] <= 0:
                    continue
                size_ratio = tri["size"] / sq["size"]
                if not (0.68 <= size_ratio <= 2.15):
                    continue
                geometry_score = marker_geometry_score(sq, tri, pose)
                min_geometry = 0.24 if tracking else 0.62
                if geometry_score < min_geometry:
                    continue
                prior_score = pose_prior_score(pose, prior_pose, frame_bgr.shape, tracking)
                min_prior = 0.25 if tracking else 0.18
                if prior_score < min_prior:
                    continue
                direction_score = triangle_direction_score(tri, pose)
                patch = warp_marker_patch(frame_bgr, pose)
                template_score = canonical_template_score(patch)
                layout_score = led_layout_score(patch)
                strip_score = strip_background_score(patch)
                min_layout = 0.12 if tracking else 0.58
                min_strip = 0.22 if tracking else 0.35
                min_template = 0.10 if tracking else 0.78
                min_square_score = 0.0 if tracking else 0.78
                if (
                    template_score < min_template
                    or layout_score < min_layout
                    or strip_score < min_strip
                    or sq["score"] < min_square_score
                ):
                    continue
                if direction_score < 0.18:
                    strong_tracking_rescue = (
                        tracking
                        and prior_score >= 0.72
                        and geometry_score >= 0.55
                        and template_score >= 0.72
                        and layout_score >= 0.50
                        and strip_score >= 0.42
                    )
                    if not strong_tracking_rescue:
                        continue
                led_score = float(np.mean(led_scores_from_pose(frame_bgr, pose)))
                base_score = (
                    0.12 * sq["score"]
                    + 0.12 * tri["score"]
                    + 0.18 * geometry_score
                    + 0.10 * direction_score
                    + 0.18 * template_score
                    + 0.12 * layout_score
                    + 0.14 * strip_score
                    + 0.04 * min(1.0, led_score)
                )
                nn_score = -1.0
                base_score *= 0.25 + 0.75 * prior_score
                candidate = Candidate(
                    pose=pose,
                    score=base_score,
                    square_score=sq["score"],
                    triangle_score=tri["score"],
                    led_score=led_score,
                    geometry_score=geometry_score,
                    template_score=template_score,
                    layout_score=layout_score,
                    strip_score=strip_score,
                    prior_score=prior_score,
                    nn_score=nn_score,
                )
                if has_patch_model:
                    cv_candidates.append(candidate)
                elif best is None or candidate.score > best.score:
                    best = candidate
        if has_patch_model:
            top_k = 8 if tracking else 16
            nn_weight = 0.22 if tracking else 0.38
            for candidate in sorted(cv_candidates, key=lambda item: item.score, reverse=True)[:top_k]:
                patch = warp_marker_patch(frame_bgr, candidate.pose)
                nn_score = self.onnx_model.score(patch)
                score = (1.0 - nn_weight) * candidate.score + nn_weight * nn_score * (0.25 + 0.75 * candidate.prior_score)
                scored = Candidate(
                    pose=candidate.pose,
                    score=score,
                    square_score=candidate.square_score,
                    triangle_score=candidate.triangle_score,
                    led_score=candidate.led_score,
                    geometry_score=candidate.geometry_score,
                    template_score=candidate.template_score,
                    layout_score=candidate.layout_score,
                    strip_score=candidate.strip_score,
                    prior_score=candidate.prior_score,
                    nn_score=nn_score,
                )
                if best is None or scored.score > best.score:
                    best = scored
        return best

    def refine_tracking_with_nn(self, frame_bgr: np.ndarray, origin: Pose) -> tuple[Pose, float, float, float, float, float] | None:
        scorer = self.onnx_model
        if scorer is None:
            return None
        best_pose = origin
        best = nn_tracking_pose_score(frame_bgr, origin, origin, scorer)
        for step in (0.036, 0.022, 0.013):
            improved = True
            for _ in range(2):
                if not improved:
                    break
                improved = False
                ax, ay = best_pose.axis
                nx, ny = best_pose.normal
                scale = max(1.0, best_pose.scale_px)
                variants = [
                    Pose((best_pose.center[0] + scale * step * ax, best_pose.center[1] + scale * step * ay), best_pose.angle_rad, best_pose.scale_px),
                    Pose((best_pose.center[0] - scale * step * ax, best_pose.center[1] - scale * step * ay), best_pose.angle_rad, best_pose.scale_px),
                    Pose((best_pose.center[0] + scale * step * nx, best_pose.center[1] + scale * step * ny), best_pose.angle_rad, best_pose.scale_px),
                    Pose((best_pose.center[0] - scale * step * nx, best_pose.center[1] - scale * step * ny), best_pose.angle_rad, best_pose.scale_px),
                    Pose(best_pose.center, best_pose.angle_rad + step * 0.65, best_pose.scale_px),
                    Pose(best_pose.center, best_pose.angle_rad - step * 0.65, best_pose.scale_px),
                    Pose(best_pose.center, best_pose.angle_rad, best_pose.scale_px * (1.0 + step * 0.45)),
                    Pose(best_pose.center, best_pose.angle_rad, best_pose.scale_px * (1.0 - step * 0.45)),
                ]
                for variant in variants:
                    score = nn_tracking_pose_score(frame_bgr, variant, origin, scorer)
                    if score[0] > best[0] + 0.002:
                        best_pose = variant
                        best = score
                        improved = True
        score, nn_score, layout_score, strip_score, border_score, led_center_score = best
        if (
            score >= 0.66
            and nn_score >= 0.82
            and layout_score >= 0.22
            and strip_score >= 0.22
            and border_score >= 0.10
            and led_center_score >= 0.26
        ):
            return best_pose, score, nn_score, layout_score, strip_score, led_center_score
        return None


class OnnxPatchScorer:
    def __init__(self, model_path: Path):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))

    def score(self, patch_bgr: np.ndarray) -> float:
        patch = cv2.resize(patch_bgr, (ONNX_PATCH_WIDTH, ONNX_PATCH_HEIGHT), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        luma = np.clip((gray - float(gray.mean())) / (float(gray.std()) + 1e-4), -3.0, 3.0) / 3.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge / (float(np.percentile(edge, 95)) + 1e-4)
        edge = np.clip(edge, 0.0, 1.0)
        blob = np.stack([luma, edge], axis=0).astype(np.float32)[None, ...]
        self.net.setInput(blob)
        logit = float(np.reshape(self.net.forward(), -1)[0])
        return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit)))))


def nn_tracking_pose_score(frame_bgr: np.ndarray, pose: Pose, origin: Pose, scorer: OnnxPatchScorer) -> tuple[float, float, float, float, float, float]:
    patch = warp_marker_patch(frame_bgr, pose, size=(ONNX_PATCH_WIDTH, ONNX_PATCH_HEIGHT))
    nn_score = scorer.score(patch)
    layout_score = led_layout_score(patch)
    strip_score = strip_background_score(patch)
    border_score = strip_border_score(patch)
    led_center_score = led_center_alignment_score(patch)
    scale = max(1.0, origin.scale_px)
    center_delta = math.hypot(pose.center[0] - origin.center[0], pose.center[1] - origin.center[1]) / scale
    angle_delta = abs(math.atan2(math.sin(pose.angle_rad - origin.angle_rad), math.cos(pose.angle_rad - origin.angle_rad)))
    scale_delta = abs(math.log(max(1.0, pose.scale_px) / scale))
    locality = math.exp(
        -0.5
        * (
            (center_delta / 0.075) ** 2
            + (angle_delta / 0.10) ** 2
            + (scale_delta / 0.070) ** 2
        )
    )
    score = (
        0.58 * nn_score
        + 0.16 * layout_score
        + 0.10 * strip_score
        + 0.06 * border_score
        + 0.10 * led_center_score
    ) * float(locality)
    return float(score), float(nn_score), float(layout_score), float(strip_score), float(border_score), float(led_center_score)


def default_tracker_model_path() -> Path | None:
    model_dir = Path(__file__).resolve().parents[2] / "models" / "tracker_likelihood"
    onnx_models = sorted(model_dir.glob("tracker_likelihood_fast_v*.onnx"))
    return onnx_models[-1] if onnx_models else None


def refine_pose(frame_bgr: np.ndarray, pose: Pose) -> RefinedPose:
    origin = pose
    (
        best_pose,
        best_score,
        best_template,
        best_layout,
        best_strip,
        best_border,
        best_led_center,
        best_square_anchor,
        best_triangle_anchor,
    ) = score_refined_pose(frame_bgr, pose, origin)
    for step, min_delta in (
        (0.060, 0.0020),
        (0.034, 0.0015),
        (0.018, 0.0010),
        (0.010, 0.0007),
        (0.006, 0.00045),
        (0.0035, 0.00030),
    ):
        improved = True
        for _ in range(2):
            if not improved:
                break
            improved = False
            variants = []
            ax, ay = best_pose.axis
            nx, ny = best_pose.normal
            scale = max(1.0, best_pose.scale_px)
            for along, normal in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
                variants.append(
                    Pose(
                        center=(best_pose.center[0] + scale * (along * ax + normal * nx), best_pose.center[1] + scale * (along * ay + normal * ny)),
                        angle_rad=best_pose.angle_rad,
                        scale_px=best_pose.scale_px,
                    )
                )
            angle_step = step * 0.72
            variants.append(Pose(best_pose.center, best_pose.angle_rad + angle_step, best_pose.scale_px))
            variants.append(Pose(best_pose.center, best_pose.angle_rad - angle_step, best_pose.scale_px))
            variants.append(Pose(best_pose.center, best_pose.angle_rad, best_pose.scale_px * (1.0 + step * 0.55)))
            variants.append(Pose(best_pose.center, best_pose.angle_rad, best_pose.scale_px * (1.0 - step * 0.55)))
            for variant in variants:
                pose_score, score, template_score, layout_score, strip_score, border_score, led_center, square_anchor, triangle_anchor = score_refined_pose(frame_bgr, variant, origin)
                if score > best_score + min_delta:
                    best_pose = pose_score
                    best_score = score
                    best_template = template_score
                    best_layout = layout_score
                    best_strip = strip_score
                    best_border = border_score
                    best_led_center = led_center
                    best_square_anchor = square_anchor
                    best_triangle_anchor = triangle_anchor
                    improved = True
    return RefinedPose(best_pose, best_score, best_template, best_layout, best_strip, best_border, best_led_center, best_square_anchor, best_triangle_anchor)


def refine_pose_near(frame_bgr: np.ndarray, origin: Pose) -> RefinedPose:
    best = refine_pose(frame_bgr, origin)
    ax, ay = origin.axis
    nx, ny = origin.normal
    scale = max(1.0, origin.scale_px)
    starts: list[Pose] = []
    for along, normal in (
        (0.0, -0.16),
        (0.0, 0.16),
        (-0.10, 0.0),
        (0.10, 0.0),
    ):
        starts.append(
            Pose(
                center=(origin.center[0] + scale * (along * ax + normal * nx), origin.center[1] + scale * (along * ay + normal * ny)),
                angle_rad=origin.angle_rad,
                scale_px=origin.scale_px,
            )
        )
    for start in starts:
        refined = refine_pose(frame_bgr, start)
        if refined.score > best.score:
            best = refined
    return best


def score_refined_pose(frame_bgr: np.ndarray, pose: Pose, origin: Pose | None = None) -> tuple[Pose, float, float, float, float, float, float, float, float]:
    patch = warp_marker_patch(frame_bgr, pose, size=REFINE_PATCH_SIZE)
    template_score = canonical_template_score(patch)
    anchor_score, square_anchor, triangle_anchor = marker_anchor_score(patch)
    layout_score = led_layout_score(patch)
    strip_score = strip_background_score(patch)
    border_score = strip_border_score(patch)
    led_center_score = led_center_alignment_score(patch)
    score = (
        0.44 * anchor_score
        + 0.14 * template_score
        + 0.10 * layout_score
        + 0.08 * strip_score
        + 0.14 * border_score
        + 0.10 * led_center_score
    )
    if origin is not None:
        score *= refine_locality_score(origin, pose)
    return pose, score, template_score, layout_score, strip_score, border_score, led_center_score, square_anchor, triangle_anchor


def refine_locality_score(origin: Pose, pose: Pose) -> float:
    scale = max(1.0, origin.scale_px)
    center_delta = math.hypot(pose.center[0] - origin.center[0], pose.center[1] - origin.center[1]) / scale
    angle_delta = math.atan2(math.sin(pose.angle_rad - origin.angle_rad), math.cos(pose.angle_rad - origin.angle_rad))
    scale_delta = abs(math.log(max(1.0, pose.scale_px) / scale))
    penalty = (center_delta / 0.13) ** 2 + (angle_delta / 0.20) ** 2 + (scale_delta / 0.13) ** 2
    return float(0.72 + 0.28 * math.exp(-0.5 * penalty))


def tracking_pose_step_plausible(pose: Pose, predicted_pose: Pose) -> bool:
    scale = max(1.0, predicted_pose.scale_px)
    center_delta = math.hypot(pose.center[0] - predicted_pose.center[0], pose.center[1] - predicted_pose.center[1]) / scale
    angle_delta = abs(math.atan2(math.sin(pose.angle_rad - predicted_pose.angle_rad), math.cos(pose.angle_rad - predicted_pose.angle_rad)))
    scale_delta = abs(math.log(max(1.0, pose.scale_px) / scale))
    return center_delta <= 0.22 and angle_delta <= 0.26 and scale_delta <= 0.14


def tracking_refine_quality(refined: RefinedPose, predicted_pose: Pose | None) -> str:
    if predicted_pose is not None and not tracking_pose_step_plausible(refined.pose, predicted_pose):
        return "none"
    strict_marker = (
        refined.score >= 0.46
        and refined.template_score >= 0.48
        and refined.layout_score >= 0.22
        and refined.strip_score >= 0.12
        and refined.border_score >= 0.10
        and refined.square_anchor_score >= 0.36
        and refined.triangle_anchor_score >= 0.36
    )
    glare_marker = (
        refined.score >= 0.58
        and refined.template_score >= 0.72
        and refined.layout_score >= 0.60
        and refined.strip_score >= 0.22
        and refined.border_score >= 0.055
        and refined.led_center_score >= 0.74
        and refined.square_anchor_score >= 0.58
        and refined.triangle_anchor_score >= 0.58
    )
    one_anchor_marker = (
        refined.score >= 0.58
        and refined.template_score >= 0.68
        and refined.layout_score >= 0.58
        and refined.strip_score >= 0.22
        and refined.border_score >= 0.24
        and refined.led_center_score >= 0.70
        and min(refined.square_anchor_score, refined.triangle_anchor_score) >= 0.30
        and max(refined.square_anchor_score, refined.triangle_anchor_score) >= 0.90
    )
    strong_shape_marker = (
        refined.score >= 0.66
        and refined.template_score >= 0.82
        and refined.layout_score >= 0.74
        and refined.strip_score >= 0.18
        and refined.border_score >= 0.030
        and refined.led_center_score >= 0.80
        and refined.square_anchor_score >= 0.86
        and refined.triangle_anchor_score >= 0.86
    )
    if strict_marker or glare_marker or one_anchor_marker or strong_shape_marker:
        return "strict"
    if predicted_pose is None:
        return "none"
    scale = max(1.0, predicted_pose.scale_px)
    center_delta = math.hypot(refined.pose.center[0] - predicted_pose.center[0], refined.pose.center[1] - predicted_pose.center[1]) / scale
    angle_delta = abs(math.atan2(math.sin(refined.pose.angle_rad - predicted_pose.angle_rad), math.cos(refined.pose.angle_rad - predicted_pose.angle_rad)))
    scale_delta = abs(math.log(max(1.0, refined.pose.scale_px) / scale))
    if center_delta > 0.08 or angle_delta > 0.10 or scale_delta > 0.075:
        return "none"
    return "none"


def tracking_refine_ok(refined: RefinedPose, predicted_pose: Pose | None) -> bool:
    return tracking_refine_quality(refined, predicted_pose) != "none"


def find_marker_shapes(frame_bgr: np.ndarray) -> tuple[list[dict[str, float | tuple[float, float]]], list[dict[str, float | tuple[float, float]]]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholds = sorted(
        {
            132,
            155,
            178,
            int(np.clip(np.percentile(blur, 94.0), 122, 205)),
            int(np.clip(np.percentile(blur, 97.0), 140, 225)),
        }
    )
    contours = []
    kernel = np.ones((3, 3), np.uint8)
    for threshold in thresholds:
        _, mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    _, mask = cv2.threshold(blur, thresholds[-1], 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.extend(found)
    squares: list[dict[str, float | tuple[float, float]]] = []
    triangles: list[dict[str, float | tuple[float, float]]] = []
    seen_squares: set[tuple[int, int, int]] = set()
    seen_triangles: set[tuple[int, int, int]] = set()
    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 18.0 or area > frame_area * 0.02:
            continue
        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.045 * peri, True)
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        size = float(max(rw, rh))
        if size <= 0:
            continue
        extent = area / max(1.0, rw * rh)
        vertices = len(approx)
        if vertices == 4:
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                continue
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            aspect = max(rw, rh) / max(1.0, min(rw, rh))
            if aspect <= 1.45 and extent >= 0.45:
                key = (round(cx / 3), round(cy / 3), round(size / 3))
                if key in seen_squares:
                    continue
                seen_squares.add(key)
                score = float(np.clip(0.45 + 0.35 * extent + 0.20 * (1.45 - aspect) / 0.45, 0.0, 1.0))
                squares.append({"center": (cx, cy), "score": score, "size": size})
        elif vertices == 3:
            points = approx.reshape(-1, 2).astype(np.float32)
            center, direction, visible_size = triangle_marker_geometry(points)
            cx, cy = center
            size = visible_size
            key = (round(cx / 3), round(cy / 3), round(size / 3))
            if key in seen_triangles:
                continue
            seen_triangles.add(key)
            extent_score = np.clip((extent - 0.25) / 0.45, 0.0, 1.0)
            score = float(np.clip(0.50 + 0.50 * extent_score, 0.0, 1.0))
            triangles.append({"center": (cx, cy), "score": score, "size": size, "direction": direction})
    squares.sort(key=lambda item: float(item["score"]), reverse=True)
    triangles.sort(key=lambda item: float(item["score"]), reverse=True)
    return squares[:32], triangles[:32]


def pose_prior_score(pose: Pose, prior_pose: Pose | None, shape: tuple[int, ...], tracking: bool) -> float:
    h, w = shape[:2]
    if prior_pose is None:
        cx, cy = w * 0.5, h * 0.5
        center_sigma = min(w, h) * 0.42
        expected_scale = min(w, h) * 0.44
        scale_sigma = 0.65
        dx = pose.center[0] - cx
        dy = pose.center[1] - cy
        center_score = math.exp(-(dx * dx + dy * dy) / (2.0 * center_sigma * center_sigma))
        scale_score = log_gaussian_ratio(pose.scale_px, expected_scale, scale_sigma)
        return float(np.clip(0.74 * center_score + 0.26 * scale_score, 0.0, 1.0))

    center_sigma = min(w, h) * (0.22 if tracking else 0.38)
    scale_sigma = 0.32 if tracking else 0.55
    angle_sigma = 0.42 if tracking else 0.95
    dx = pose.center[0] - prior_pose.center[0]
    dy = pose.center[1] - prior_pose.center[1]
    center_score = math.exp(-(dx * dx + dy * dy) / (2.0 * center_sigma * center_sigma))
    scale_score = log_gaussian_ratio(pose.scale_px, prior_pose.scale_px, scale_sigma)
    angle_delta = math.atan2(math.sin(pose.angle_rad - prior_pose.angle_rad), math.cos(pose.angle_rad - prior_pose.angle_rad))
    angle_score = math.exp(-(angle_delta * angle_delta) / (2.0 * angle_sigma * angle_sigma))
    if tracking:
        pose_score = 0.62 * center_score + 0.38 * scale_score
        return float(np.clip(pose_score * (0.18 + 0.82 * angle_score), 0.0, 1.0))
    return float(np.clip(0.58 * center_score + 0.24 * scale_score + 0.18 * angle_score, 0.0, 1.0))


def marker_geometry_score(sq: dict[str, object], tri: dict[str, object], pose: Pose) -> float:
    square_norm = float(sq["size"]) / max(1.0, pose.scale_px)
    triangle_norm = float(tri["size"]) / max(1.0, pose.scale_px)
    square_fit = log_gaussian_ratio(square_norm, SQUARE_SIZE_FRACTION, sigma=0.42)
    triangle_fit = log_gaussian_ratio(triangle_norm, TRIANGLE_SIZE_FRACTION, sigma=0.45)
    ratio_fit = log_gaussian_ratio(float(tri["size"]) / max(1.0, float(sq["size"])), TRIANGLE_SIZE_FRACTION / SQUARE_SIZE_FRACTION, sigma=0.38)
    return float(np.clip(0.35 * square_fit + 0.35 * triangle_fit + 0.30 * ratio_fit, 0.0, 1.0))


def log_gaussian_ratio(value: float, expected: float, sigma: float) -> float:
    if value <= 0.0 or expected <= 0.0:
        return 0.0
    error = math.log(value / expected)
    return float(math.exp(-(error * error) / (2.0 * sigma * sigma)))


def triangle_direction_score(triangle: dict[str, object], pose: Pose) -> float:
    direction = triangle.get("direction")
    if direction is None:
        return 0.5
    dx, dy = direction  # type: ignore[misc]
    nx, ny = pose.normal
    dot = float(dx) * (-nx) + float(dy) * (-ny)
    return float(np.clip((dot + 0.20) / 1.20, 0.0, 1.0))


def triangle_marker_geometry(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float]:
    if len(points) != 3:
        return (0.0, 0.0), (0.0, -1.0), 0.0
    lengths = []
    for index in range(3):
        a = points[index]
        b = points[(index + 1) % 3]
        lengths.append(float(np.linalg.norm(a - b)))
    base_index = int(np.argmin(lengths))
    base_a = points[base_index]
    base_b = points[(base_index + 1) % 3]
    apex_index = (base_index + 2) % 3
    apex = points[apex_index]
    base_mid = (base_a + base_b) * 0.5
    center = (apex + base_mid) * 0.5
    vector = apex - center
    norm = float(np.linalg.norm(vector))
    if norm < 1e-4:
        direction = (0.0, -1.0)
    else:
        direction = (float(vector[0] / norm), float(vector[1] / norm))
    height = float(np.linalg.norm(apex - base_mid))
    base = float(np.linalg.norm(base_a - base_b))
    return (float(center[0]), float(center[1])), direction, max(height, base)


def canonical_template_score(patch_bgr: np.ndarray) -> float:
    height, width = patch_bgr.shape[:2]
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bgr = patch_bgr.astype(np.float32) / 255.0
    xs = (np.arange(width, dtype=np.float32) / max(1, width - 1) - 0.5) * 1.35
    ys = (np.arange(height, dtype=np.float32) / max(1, height - 1) - 0.5) * 0.46
    grid_x, grid_y = np.meshgrid(xs, ys)

    square_half = SQUARE_SIZE_FRACTION * 0.5
    square_inside = (np.abs(grid_x + 0.5) <= square_half) & (np.abs(grid_y) <= square_half)
    square_outer = (np.abs(grid_x + 0.5) <= square_half * 2.0) & (np.abs(grid_y) <= square_half * 2.0) & ~square_inside

    triangle_half = TRIANGLE_SIZE_FRACTION * 0.5
    triangle_inside = triangle_mask(grid_x, grid_y, triangle_half)
    triangle_outer = (
        (grid_x >= 0.5 - triangle_half * 1.45)
        & (grid_x <= 0.5 + triangle_half * 1.45)
        & (grid_y >= -triangle_half * 1.35)
        & (grid_y <= triangle_half * 1.55)
        & ~triangle_inside
    )

    led_mask = np.zeros_like(square_inside)
    led_radius = LED_RADIUS_FRACTION * 0.78
    for fraction in LED_FRACTIONS:
        cx = float(fraction) - 0.5
        led_mask |= (grid_x - cx) ** 2 + grid_y**2 <= led_radius**2

    square_score = contrast_score(gray, square_inside, square_outer, margin=0.06)
    triangle_score = contrast_score(gray, triangle_inside, triangle_outer, margin=0.06)
    neutral = marker_neutrality_score(bgr, square_inside | triangle_inside)
    led_line = float(np.clip((float(gray[led_mask].mean()) - float(gray[~led_mask].mean()) + 0.18) / 0.45, 0.0, 1.0)) if led_mask.any() else 0.0
    return float(np.clip(0.34 * square_score + 0.34 * triangle_score + 0.18 * neutral + 0.14 * led_line, 0.0, 1.0))


def marker_anchor_score(patch_bgr: np.ndarray) -> tuple[float, float, float]:
    height, width = patch_bgr.shape[:2]
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bgr = patch_bgr.astype(np.float32) / 255.0
    xs = (np.arange(width, dtype=np.float32) / max(1, width - 1) - 0.5) * 1.35
    ys = (np.arange(height, dtype=np.float32) / max(1, height - 1) - 0.5) * 0.46
    grid_x, grid_y = np.meshgrid(xs, ys)

    square_half = SQUARE_SIZE_FRACTION * 0.5
    square_inside = (np.abs(grid_x + 0.5) <= square_half) & (np.abs(grid_y) <= square_half)
    square_outer = (np.abs(grid_x + 0.5) <= square_half * 2.0) & (np.abs(grid_y) <= square_half * 2.0) & ~square_inside

    triangle_half = TRIANGLE_SIZE_FRACTION * 0.5
    triangle_inside = triangle_mask(grid_x, grid_y, triangle_half)
    triangle_outer = (
        (grid_x >= 0.5 - triangle_half * 1.45)
        & (grid_x <= 0.5 + triangle_half * 1.45)
        & (grid_y >= -triangle_half * 1.35)
        & (grid_y <= triangle_half * 1.55)
        & ~triangle_inside
    )

    square_contrast = contrast_score(gray, square_inside, square_outer, margin=0.06)
    triangle_contrast = contrast_score(gray, triangle_inside, triangle_outer, margin=0.06)
    square_position = bright_centroid_score(
        gray,
        grid_x,
        grid_y,
        center_x=-0.5,
        center_y=0.0,
        window_half_x=square_half * 2.25,
        window_half_y=square_half * 2.25,
        sigma=SQUARE_SIZE_FRACTION * 0.23,
        baseline_percentile=62.0,
        compact_sigma=SQUARE_SIZE_FRACTION * 0.42,
    )
    triangle_position = bright_centroid_score(
        gray,
        grid_x,
        grid_y,
        center_x=0.5,
        center_y=triangle_half / 3.0,
        window_half_x=triangle_half * 2.45,
        window_half_y=triangle_half * 2.45,
        sigma=TRIANGLE_SIZE_FRACTION * 0.28,
        baseline_percentile=62.0,
        compact_sigma=TRIANGLE_SIZE_FRACTION * 0.46,
    )
    square_score = float(np.clip(0.62 * square_contrast + 0.38 * square_position, 0.0, 1.0))
    triangle_score = float(np.clip(0.62 * triangle_contrast + 0.38 * triangle_position, 0.0, 1.0))
    neutral = marker_neutrality_score(bgr, square_inside | triangle_inside)
    anchor_score = float(np.clip(0.42 * square_score + 0.42 * triangle_score + 0.16 * neutral, 0.0, 1.0))
    return anchor_score, float(square_score), float(triangle_score)


def bright_centroid_score(
    gray: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    window_half_x: float,
    window_half_y: float,
    sigma: float,
    baseline_percentile: float,
    compact_sigma: float,
) -> float:
    window = (np.abs(grid_x - center_x) <= window_half_x) & (np.abs(grid_y - center_y) <= window_half_y)
    if not window.any():
        return 0.0
    values = gray[window]
    activation = values - float(np.percentile(values, baseline_percentile))
    activation = np.clip(activation, 0.0, None)
    total = float(activation.sum())
    if total < 1e-4:
        return 0.0
    wx = grid_x[window]
    wy = grid_y[window]
    centroid_x = float((activation * wx).sum() / total)
    centroid_y = float((activation * wy).sum() / total)
    error = math.hypot(centroid_x - center_x, centroid_y - center_y)
    centered_x = wx - centroid_x
    centered_y = wy - centroid_y
    rms = math.sqrt(float((activation * (centered_x * centered_x + centered_y * centered_y)).sum() / total))
    centroid_score = math.exp(-0.5 * (error / max(1e-4, sigma)) ** 2)
    compact_score = math.exp(-0.5 * (rms / max(1e-4, compact_sigma)) ** 2)
    return float(np.clip(0.72 * centroid_score + 0.28 * compact_score, 0.0, 1.0))


def led_layout_score(patch_bgr: np.ndarray) -> float:
    height, width = patch_bgr.shape[:2]
    bgr = patch_bgr.astype(np.float32) / 255.0
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blueish = np.clip(bgr[:, :, 0] - 0.5 * bgr[:, :, 1] - 0.5 * bgr[:, :, 2], -1.0, 1.0)
    xs = (np.arange(width, dtype=np.float32) / max(1, width - 1) - 0.5) * 1.35
    ys = (np.arange(height, dtype=np.float32) / max(1, height - 1) - 0.5) * 0.46
    grid_x, grid_y = np.meshgrid(xs, ys)

    scores = []
    radius = LED_RADIUS_FRACTION * 0.95
    for fraction in LED_FRACTIONS:
        cx = float(fraction) - 0.5
        dist2 = (grid_x - cx) ** 2 + grid_y**2
        center = dist2 <= (radius * 0.95) ** 2
        ring = (dist2 >= (radius * 1.35) ** 2) & (dist2 <= (radius * 2.25) ** 2)
        if not center.any() or not ring.any():
            scores.append(0.0)
            continue
        center_bright = float(np.percentile(gray[center], 65))
        ring_bright = float(np.percentile(gray[ring], 50))
        center_blue = float(np.percentile(blueish[center], 65))
        compact = np.clip((center_bright - ring_bright + 0.06) / 0.34, 0.0, 1.0)
        color = np.clip((center_blue + 0.08) / 0.22, 0.0, 1.0)
        visible = np.clip((center_bright - 0.10) / 0.42, 0.0, 1.0)
        scores.append(float(np.clip(0.48 * compact + 0.34 * color + 0.18 * visible, 0.0, 1.0)))
    if not scores:
        return 0.0
    scores_array = np.array(scores, dtype=np.float32)
    mean_score = float(scores_array.mean())
    weak_score = float(np.percentile(scores_array, 25))
    return float(np.clip(0.65 * mean_score + 0.35 * weak_score, 0.0, 1.0))


def led_center_alignment_score(patch_bgr: np.ndarray) -> float:
    height, width = patch_bgr.shape[:2]
    bgr = patch_bgr.astype(np.float32) / 255.0
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blueish = np.clip(bgr[:, :, 0] - 0.5 * bgr[:, :, 1] - 0.5 * bgr[:, :, 2], 0.0, 1.0)
    xs = (np.arange(width, dtype=np.float32) / max(1, width - 1) - 0.5) * 1.35
    ys = (np.arange(height, dtype=np.float32) / max(1, height - 1) - 0.5) * 0.46
    grid_x, grid_y = np.meshgrid(xs, ys)

    scores = []
    radius = LED_RADIUS_FRACTION * 2.55
    target_radius = max(LED_RADIUS_FRACTION * 0.80, 1e-4)
    for fraction in LED_FRACTIONS:
        cx = float(fraction) - 0.5
        dist2 = (grid_x - cx) ** 2 + grid_y**2
        local = dist2 <= radius**2
        if not local.any():
            continue
        local_gray = gray[local]
        local_blue = blueish[local]
        activation = 0.58 * local_blue + 0.42 * local_gray
        activation = activation - float(np.percentile(activation, 28))
        activation = np.clip(activation, 0.0, None)
        total = float(activation.sum())
        if total < 0.015:
            continue
        lx = grid_x[local]
        ly = grid_y[local]
        centroid_x = float((activation * lx).sum() / total)
        centroid_y = float((activation * ly).sum() / total)
        error = math.hypot(centroid_x - cx, centroid_y)
        shape = float(np.clip((float(np.percentile(local_gray, 70)) - float(np.percentile(local_gray, 30)) + 0.08) / 0.34, 0.0, 1.0))
        position = math.exp(-0.5 * (error / target_radius) ** 2)
        scores.append(float(0.78 * position + 0.22 * shape))
    if not scores:
        return 0.0
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def strip_background_score(patch_bgr: np.ndarray) -> float:
    height, width = patch_bgr.shape[:2]
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    xs = (np.arange(width, dtype=np.float32) / max(1, width - 1) - 0.5) * 1.35
    ys = (np.arange(height, dtype=np.float32) / max(1, height - 1) - 0.5) * 0.46
    grid_x, grid_y = np.meshgrid(xs, ys)

    inner = (np.abs(grid_x) <= 0.64) & (np.abs(grid_y) <= 0.145)
    outer = (
        ((np.abs(grid_x) <= 0.64) & (np.abs(grid_y) >= 0.185) & (np.abs(grid_y) <= 0.225))
        | ((np.abs(grid_x) >= 0.72) & (np.abs(grid_x) <= 0.96) & (np.abs(grid_y) <= 0.20))
    )

    keepout = np.zeros_like(inner)
    square_half = SQUARE_SIZE_FRACTION * 0.75
    keepout |= (np.abs(grid_x + 0.5) <= square_half) & (np.abs(grid_y) <= square_half)
    triangle_half = TRIANGLE_SIZE_FRACTION * 0.62
    keepout |= (
        (grid_x >= 0.5 - triangle_half * 1.45)
        & (grid_x <= 0.5 + triangle_half * 1.45)
        & (grid_y >= -triangle_half * 1.45)
        & (grid_y <= triangle_half * 1.65)
    )
    led_radius = LED_RADIUS_FRACTION * 1.05
    for fraction in LED_FRACTIONS:
        cx = float(fraction) - 0.5
        keepout |= (grid_x - cx) ** 2 + grid_y**2 <= led_radius**2

    inner_bg = inner & ~keepout
    if not inner_bg.any() or not outer.any():
        return 0.0

    inner_values = gray[inner_bg]
    outer_values = gray[outer]
    inner_mean = float(np.percentile(inner_values, 58))
    outer_mean = float(np.percentile(outer_values, 55))
    inner_std = float(np.std(inner_values))

    dark = np.clip((0.58 - inner_mean) / 0.50, 0.0, 1.0)
    contrast = np.clip((outer_mean - inner_mean + 0.04) / 0.30, 0.0, 1.0)
    calm = np.clip(1.0 - inner_std / 0.22, 0.0, 1.0)
    return float(np.clip(0.42 * dark + 0.36 * contrast + 0.22 * calm, 0.0, 1.0))


def strip_border_score(patch_bgr: np.ndarray) -> float:
    height, width = patch_bgr.shape[:2]
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    xs = (np.arange(width, dtype=np.float32) / max(1, width - 1) - 0.5) * 1.35
    ys = (np.arange(height, dtype=np.float32) / max(1, height - 1) - 0.5) * 0.46
    grid_x, grid_y = np.meshgrid(xs, ys)

    x_edge = 0.67
    y_edge = 0.185
    thickness = 0.020
    border = (
        ((np.abs(grid_x) <= x_edge) & (np.abs(np.abs(grid_y) - y_edge) <= thickness))
        | ((np.abs(np.abs(grid_x) - x_edge) <= thickness) & (np.abs(grid_y) <= y_edge))
    )
    inside = (np.abs(grid_x) <= x_edge - 0.055) & (np.abs(grid_y) <= y_edge - 0.055)
    outside = (
        ((np.abs(grid_x) <= x_edge + 0.055) & (np.abs(np.abs(grid_y) - (y_edge + 0.050)) <= thickness))
        | ((np.abs(np.abs(grid_x) - (x_edge + 0.050)) <= thickness) & (np.abs(grid_y) <= y_edge + 0.050))
    )
    if not border.any() or not inside.any() or not outside.any():
        return 0.0
    border_value = float(np.percentile(gray[border], 70))
    inside_value = float(np.percentile(gray[inside], 55))
    outside_value = float(np.percentile(gray[outside], 55))
    contrast = np.clip((border_value - max(inside_value, outside_value) + 0.06) / 0.30, 0.0, 1.0)
    dark_inside = np.clip((0.58 - inside_value) / 0.50, 0.0, 1.0)
    return float(np.clip(0.70 * contrast + 0.30 * dark_inside, 0.0, 1.0))


def triangle_mask(grid_x: np.ndarray, grid_y: np.ndarray, half: float) -> np.ndarray:
    ax, ay = 0.5, -half
    bx, by = 0.5 + half, half
    cx, cy = 0.5 - half, half
    denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    a = ((by - cy) * (grid_x - cx) + (cx - bx) * (grid_y - cy)) / denominator
    b = ((cy - ay) * (grid_x - cx) + (ax - cx) * (grid_y - cy)) / denominator
    c = 1.0 - a - b
    return (a >= 0.0) & (b >= 0.0) & (c >= 0.0)


def contrast_score(gray: np.ndarray, inside: np.ndarray, outside: np.ndarray, margin: float) -> float:
    if not inside.any() or not outside.any():
        return 0.0
    delta = float(np.percentile(gray[inside], 65) - np.percentile(gray[outside], 55))
    return float(np.clip((delta - margin) / 0.35, 0.0, 1.0))


def marker_neutrality_score(bgr: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    values = bgr[mask]
    spread = float(np.mean(np.max(values, axis=1) - np.min(values, axis=1)))
    return float(np.clip(1.0 - spread / 0.32, 0.0, 1.0))


class CvTrackerBackend:
    def __init__(self, tracker_model: Path | None = None, accept_score: float = 0.48):
        self.detector = CvMarkerDetector(tracker_model)
        self.accept_score = accept_score
        self.acquire_accept_score = max(0.64, accept_score)
        self.refine_accept_score = 0.50
        self.max_tracking_misses = 12
        self.max_lost_prior_frames = 180
        self.was_tracking = False
        self.last_pose: Pose | None = None
        self.last_real_pose: Pose | None = None
        self.last_real_frame = 0
        self.frame_index = 0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_angle = 0.0
        self.velocity_log_scale = 0.0
        self.missed_frames = 0
        self.established_frames = 0
        self.flow_gray: np.ndarray | None = None
        self.flow_pose: Pose | None = None

    def reset(self) -> None:
        self.was_tracking = False
        self.last_pose = None
        self.last_real_pose = None
        self.last_real_frame = 0
        self.frame_index = 0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_angle = 0.0
        self.velocity_log_scale = 0.0
        self.missed_frames = 0
        self.established_frames = 0
        self.flow_gray = None
        self.flow_pose = None

    def process(self, frame_bgr: np.ndarray, timestamp_ns: int) -> TrackerResult:
        self.frame_index += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        predicted_pose = self._predicted_pose() if self.was_tracking else (self.last_pose if self.missed_frames <= 8 else None)
        candidate = self.detector.detect(frame_bgr, predicted_pose, self.was_tracking)
        threshold = self.accept_score if self.was_tracking else self.acquire_accept_score
        if self.was_tracking and candidate is not None and candidate.prior_score < 0.55:
            candidate = None
        if self.was_tracking and candidate is not None and predicted_pose is not None and not tracking_pose_step_plausible(candidate.pose, predicted_pose):
            candidate = None
        if candidate is None or candidate.score < threshold:
            h, w = frame_bgr.shape[:2]
            recovery_pose = predicted_pose if predicted_pose is not None else self.last_pose
            if not self.was_tracking and self.frame_index % 4 == 1:
                soft_acquired = self._try_soft_acquire(frame_bgr, gray)
                if soft_acquired is not None:
                    return soft_acquired
            if self.was_tracking and self.missed_frames >= 1:
                reacquired = self._try_reacquire(frame_bgr, gray)
                if reacquired is not None:
                    return reacquired
            if self.was_tracking and recovery_pose is not None:
                nn_refined = self.detector.refine_tracking_with_nn(frame_bgr, recovery_pose)
                if nn_refined is not None:
                    nn_pose, nn_score, nn_like, nn_layout, nn_strip, nn_led_center = nn_refined
                    refined = refine_pose(frame_bgr, nn_pose)
                    result = self._tracking_refine_result(
                        frame_bgr,
                        gray,
                        refined,
                        predicted_pose,
                        "track_nn",
                        (
                            f"cv_nn score={nn_score:.2f} nn={nn_like:.2f} "
                            f"layout={nn_layout:.2f} strip={nn_strip:.2f} ledC={nn_led_center:.2f} "
                            f"ref={refined.score:.2f}/{refined.template_score:.2f}/"
                            f"{refined.layout_score:.2f}/{refined.strip_score:.2f}/"
                            f"{refined.border_score:.2f}/{refined.led_center_score:.2f}/"
                            f"{refined.square_anchor_score:.2f}/{refined.triangle_anchor_score:.2f}"
                        ),
                    )
                    if result is not None:
                        return result
            flow_pose = self._flow_pose(gray)
            if flow_pose is not None:
                refined = refine_pose(frame_bgr, flow_pose)
                result = self._tracking_refine_result(
                    frame_bgr,
                    gray,
                    refined,
                    predicted_pose,
                    "track_flow",
                    (
                        f"cv_flow tmpl={refined.template_score:.2f} layout={refined.layout_score:.2f} "
                        f"strip={refined.strip_score:.2f} border={refined.border_score:.2f} "
                        f"ledC={refined.led_center_score:.2f} "
                        f"sqA={refined.square_anchor_score:.2f} triA={refined.triangle_anchor_score:.2f}"
                    ),
                )
                if result is not None:
                    return result
            if recovery_pose is not None and self.missed_frames < self.max_lost_prior_frames:
                refined = refine_pose(frame_bgr, recovery_pose)
                if self.was_tracking:
                    marker_refine_ok = tracking_refine_quality(refined, predicted_pose) != "none"
                else:
                    marker_refine_ok = (
                        refined.score >= 0.56
                        and (
                            (refined.template_score >= 0.50 and refined.layout_score >= 0.55)
                            or (refined.template_score >= 0.72 and refined.layout_score >= 0.35)
                        )
                        and refined.strip_score >= 0.18
                        and refined.border_score >= 0.30
                        and refined.led_center_score >= 0.40
                        and refined.square_anchor_score >= 0.46
                        and refined.triangle_anchor_score >= 0.46
                    )
                if marker_refine_ok:
                    if self.was_tracking:
                        result = self._tracking_refine_result(
                            frame_bgr,
                            gray,
                            refined,
                            predicted_pose,
                            "track_refine",
                            (
                                f"cv_refine tmpl={refined.template_score:.2f} layout={refined.layout_score:.2f} "
                                f"strip={refined.strip_score:.2f} border={refined.border_score:.2f} "
                                f"ledC={refined.led_center_score:.2f} "
                                f"sqA={refined.square_anchor_score:.2f} triA={refined.triangle_anchor_score:.2f}"
                            ),
                        )
                        if result is not None:
                            return result
                    else:
                        self.was_tracking = True
                        self._accept_pose(refined.pose, gray)
                        self.missed_frames = 0
                        self.established_frames += 1
                        return self._pose_result(
                            frame_bgr,
                            refined.pose,
                            "track_recover",
                            refined.score,
                            (
                                f"cv_refine tmpl={refined.template_score:.2f} layout={refined.layout_score:.2f} "
                                f"strip={refined.strip_score:.2f} border={refined.border_score:.2f} "
                                f"ledC={refined.led_center_score:.2f} "
                                f"sqA={refined.square_anchor_score:.2f} triA={refined.triangle_anchor_score:.2f}"
                            ),
                        )
                refine_fail_debug = (
                    f" ref_fail={refined.score:.2f}/{refined.template_score:.2f}/"
                    f"{refined.layout_score:.2f}/{refined.strip_score:.2f}/"
                    f"{refined.border_score:.2f}/{refined.led_center_score:.2f}/"
                    f"{refined.square_anchor_score:.2f}/{refined.triangle_anchor_score:.2f}"
                )
            else:
                refine_fail_debug = ""
            if self.was_tracking and predicted_pose is not None and self.missed_frames < self.max_tracking_misses:
                near_refined = refine_pose_near(frame_bgr, predicted_pose)
                result = self._tracking_refine_result(
                    frame_bgr,
                    gray,
                    near_refined,
                    predicted_pose,
                    "track_near",
                    (
                        f"cv_near tmpl={near_refined.template_score:.2f} layout={near_refined.layout_score:.2f} "
                        f"strip={near_refined.strip_score:.2f} border={near_refined.border_score:.2f} "
                        f"ledC={near_refined.led_center_score:.2f} "
                        f"sqA={near_refined.square_anchor_score:.2f} triA={near_refined.triangle_anchor_score:.2f}"
                    ),
                )
                if result is not None:
                    return result
            if (
                self.was_tracking
                and predicted_pose is not None
                and self.established_frames >= 4
                and self.missed_frames < self.max_tracking_misses
            ):
                self.missed_frames += 1
                self.last_pose = predicted_pose
                return self._hold_result(frame_bgr, predicted_pose, f"cv_predict missed={self.missed_frames}{refine_fail_debug}")
            self.was_tracking = False
            self.established_frames = 0
            if self.last_pose is not None:
                self.missed_frames += 1
                if self.missed_frames > self.max_lost_prior_frames:
                    self.last_pose = None
                    self.last_real_pose = None
                    self.velocity_x = 0.0
                    self.velocity_y = 0.0
                    self.velocity_angle = 0.0
                    self.velocity_log_scale = 0.0
                    self.missed_frames = 0
            else:
                self.missed_frames = 0
            return TrackerResult(hit=False, mode="acquire", x=w * 0.5, y=h * 0.5, debug="cv_no_candidate")
        pose = candidate.pose
        mode = "track" if self.was_tracking else "acquire"
        refined = refine_pose(frame_bgr, pose)
        refined_ok = (
            refined.score >= (0.42 if mode == "track" else 0.48)
            and refined.border_score >= (0.10 if mode == "track" else 0.28)
            and refined.led_center_score >= (0.10 if mode == "track" else 0.34)
            and refined.square_anchor_score >= (0.32 if mode == "track" else 0.42)
            and refined.triangle_anchor_score >= (0.32 if mode == "track" else 0.42)
            and (
                mode != "track"
                or predicted_pose is None
                or tracking_pose_step_plausible(refined.pose, predicted_pose)
            )
        )
        if mode == "track" and not refined_ok:
            recovery_pose = predicted_pose if predicted_pose is not None else self.last_pose
            if recovery_pose is not None:
                nn_refined = self.detector.refine_tracking_with_nn(frame_bgr, recovery_pose)
                if nn_refined is not None:
                    nn_pose, nn_score, nn_like, nn_layout, nn_strip, nn_led_center = nn_refined
                    refined = refine_pose(frame_bgr, nn_pose)
                    result = self._tracking_refine_result(
                        frame_bgr,
                        gray,
                        refined,
                        predicted_pose,
                        "track_nn",
                        (
                            f"cv_nn_after_reject score={nn_score:.2f} nn={nn_like:.2f} "
                            f"layout={nn_layout:.2f} strip={nn_strip:.2f} ledC={nn_led_center:.2f} "
                            f"ref={refined.score:.2f}/{refined.template_score:.2f}/"
                            f"{refined.layout_score:.2f}/{refined.strip_score:.2f}/"
                            f"{refined.border_score:.2f}/{refined.led_center_score:.2f}/"
                            f"{refined.square_anchor_score:.2f}/{refined.triangle_anchor_score:.2f}"
                        ),
                    )
                    if result is not None:
                        return result
            flow_pose = self._flow_pose(gray)
            if flow_pose is not None:
                flow_refined = refine_pose(frame_bgr, flow_pose)
                result = self._tracking_refine_result(
                    frame_bgr,
                    gray,
                    flow_refined,
                    predicted_pose,
                    "track_flow",
                    (
                        f"cv_flow_after_reject tmpl={flow_refined.template_score:.2f} "
                        f"layout={flow_refined.layout_score:.2f} strip={flow_refined.strip_score:.2f} "
                        f"border={flow_refined.border_score:.2f} ledC={flow_refined.led_center_score:.2f} "
                        f"sqA={flow_refined.square_anchor_score:.2f} triA={flow_refined.triangle_anchor_score:.2f}"
                    ),
                )
                if result is not None:
                    return result
            if recovery_pose is not None:
                recovery_refined = refine_pose(frame_bgr, recovery_pose)
                result = self._tracking_refine_result(
                    frame_bgr,
                    gray,
                    recovery_refined,
                    predicted_pose,
                    "track_refine",
                    (
                        f"cv_refine_after_reject tmpl={recovery_refined.template_score:.2f} "
                        f"layout={recovery_refined.layout_score:.2f} strip={recovery_refined.strip_score:.2f} "
                        f"border={recovery_refined.border_score:.2f} ledC={recovery_refined.led_center_score:.2f} "
                        f"sqA={recovery_refined.square_anchor_score:.2f} triA={recovery_refined.triangle_anchor_score:.2f}"
                    ),
                )
                if result is not None:
                    return result
            if predicted_pose is not None:
                near_refined = refine_pose_near(frame_bgr, predicted_pose)
                result = self._tracking_refine_result(
                    frame_bgr,
                    gray,
                    near_refined,
                    predicted_pose,
                    "track_near",
                    (
                        f"cv_near_after_reject tmpl={near_refined.template_score:.2f} "
                        f"layout={near_refined.layout_score:.2f} strip={near_refined.strip_score:.2f} "
                        f"border={near_refined.border_score:.2f} ledC={near_refined.led_center_score:.2f} "
                        f"sqA={near_refined.square_anchor_score:.2f} triA={near_refined.triangle_anchor_score:.2f}"
                    ),
                )
                if result is not None:
                    return result
            if self.missed_frames >= 1:
                reacquired = self._try_reacquire(frame_bgr, gray)
                if reacquired is not None:
                    return reacquired
            if self.established_frames >= 4 and predicted_pose is not None and self.missed_frames < self.max_tracking_misses:
                self.missed_frames += 1
                self.last_pose = predicted_pose
                return self._hold_result(
                    frame_bgr,
                    predicted_pose,
                    (
                        f"cv_reject_candidate missed={self.missed_frames} "
                        f"ref_fail={refined.score:.2f}/{refined.template_score:.2f}/"
                        f"{refined.layout_score:.2f}/{refined.strip_score:.2f}/"
                        f"{refined.border_score:.2f}/{refined.led_center_score:.2f}/"
                        f"{refined.square_anchor_score:.2f}/{refined.triangle_anchor_score:.2f}"
                    ),
                )
        if refined_ok:
            pose = refined.pose
        self.was_tracking = True
        self._accept_pose(pose, gray)
        self.missed_frames = 0
        self.established_frames = self.established_frames + 1 if mode == "track" else 1
        refine_debug = (
            (
                f" ref={refined.score:.2f}/{refined.template_score:.2f}/{refined.layout_score:.2f}/"
                f"{refined.strip_score:.2f}/{refined.border_score:.2f}/{refined.led_center_score:.2f}/"
                f"{refined.square_anchor_score:.2f}/{refined.triangle_anchor_score:.2f}"
            )
            if refined_ok
            else ""
        )
        return self._pose_result(
            frame_bgr,
            pose,
            mode,
            candidate.score,
            (
                f"cv sq={candidate.square_score:.2f} tri={candidate.triangle_score:.2f} "
                f"geom={candidate.geometry_score:.2f} tmpl={candidate.template_score:.2f} "
                f"layout={candidate.layout_score:.2f} strip={candidate.strip_score:.2f} "
                f"prior={candidate.prior_score:.2f} led={candidate.led_score:.2f} "
                f"nn={candidate.nn_score:.2f}{refine_debug}"
            ),
        )

    def _accept_pose(self, pose: Pose, gray: np.ndarray | None = None) -> None:
        if self.last_real_pose is not None:
            dt = max(1, self.frame_index - self.last_real_frame)
            vx = (pose.center[0] - self.last_real_pose.center[0]) / dt
            vy = (pose.center[1] - self.last_real_pose.center[1]) / dt
            angle_delta = math.atan2(math.sin(pose.angle_rad - self.last_real_pose.angle_rad), math.cos(pose.angle_rad - self.last_real_pose.angle_rad)) / dt
            scale_delta = math.log(max(1.0, pose.scale_px) / max(1.0, self.last_real_pose.scale_px)) / dt
            speed_limit = max(2.0, pose.scale_px * 0.055)
            speed = math.hypot(vx, vy)
            if speed > speed_limit:
                factor = speed_limit / speed
                vx *= factor
                vy *= factor
            angle_delta = float(np.clip(angle_delta, -0.055, 0.055))
            scale_delta = float(np.clip(scale_delta, -0.045, 0.045))
            alpha = 0.45
            self.velocity_x = (1.0 - alpha) * self.velocity_x + alpha * vx
            self.velocity_y = (1.0 - alpha) * self.velocity_y + alpha * vy
            self.velocity_angle = (1.0 - alpha) * self.velocity_angle + alpha * angle_delta
            self.velocity_log_scale = (1.0 - alpha) * self.velocity_log_scale + alpha * scale_delta
        self.last_pose = pose
        self.last_real_pose = pose
        self.last_real_frame = self.frame_index
        if gray is not None:
            self.flow_gray = gray
            self.flow_pose = pose

    def _tracking_refine_result(
        self,
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        refined: RefinedPose,
        predicted_pose: Pose | None,
        mode: str,
        debug: str,
    ) -> TrackerResult | None:
        quality = tracking_refine_quality(refined, predicted_pose)
        if quality == "strict":
            self._accept_pose(refined.pose, gray)
            self.missed_frames = 0
            self.established_frames += 1
            return self._pose_result(frame_bgr, refined.pose, mode, refined.score, debug)
        return None

    def _try_reacquire(self, frame_bgr: np.ndarray, gray: np.ndarray) -> TrackerResult | None:
        reacquire = self.detector.detect(frame_bgr, None, False)
        if reacquire is None or reacquire.score < self.acquire_accept_score:
            reacquire = self.detector.detect(frame_bgr, None, True)
        if reacquire is None or reacquire.score < 0.36:
            return None
        refined = refine_pose(frame_bgr, reacquire.pose)
        balanced_reacquire_ok = (
            refined.score >= 0.48
            and refined.template_score >= 0.72
            and refined.layout_score >= 0.58
            and refined.strip_score >= 0.24
            and refined.border_score >= 0.10
            and refined.led_center_score >= 0.34
            and refined.square_anchor_score >= 0.42
            and refined.triangle_anchor_score >= 0.42
        )
        anchored_reacquire_ok = (
            refined.score >= 0.60
            and refined.template_score >= 0.65
            and refined.layout_score >= 0.50
            and refined.strip_score >= 0.34
            and refined.border_score >= 0.28
            and refined.led_center_score >= 0.50
            and min(refined.square_anchor_score, refined.triangle_anchor_score) >= 0.30
            and max(refined.square_anchor_score, refined.triangle_anchor_score) >= 0.90
        )
        partial_anchor_reacquire_ok = (
            reacquire.square_score >= 0.75
            and reacquire.triangle_score >= 0.75
            and reacquire.geometry_score >= 0.85
            and (reacquire.nn_score < 0.0 or reacquire.nn_score >= 0.42)
            and refined.score >= 0.49
            and refined.template_score >= 0.55
            and refined.layout_score >= 0.40
            and refined.strip_score >= 0.34
            and refined.border_score >= 0.28
            and refined.led_center_score >= 0.45
            and min(refined.square_anchor_score, refined.triangle_anchor_score) >= 0.20
            and max(refined.square_anchor_score, refined.triangle_anchor_score) >= 0.75
        )
        reacquire_ok = balanced_reacquire_ok or anchored_reacquire_ok or partial_anchor_reacquire_ok
        if not reacquire_ok:
            return None
        self._accept_pose(refined.pose, gray)
        self.missed_frames = 0
        self.established_frames = max(self.established_frames, 4)
        return self._pose_result(
            frame_bgr,
            refined.pose,
            "reacquire",
            reacquire.score,
            (
                f"cv_reacquire sq={reacquire.square_score:.2f} tri={reacquire.triangle_score:.2f} "
                f"geom={reacquire.geometry_score:.2f} tmpl={reacquire.template_score:.2f} "
                f"layout={reacquire.layout_score:.2f} strip={reacquire.strip_score:.2f} "
                f"prior={reacquire.prior_score:.2f} led={reacquire.led_score:.2f} "
                f"nn={reacquire.nn_score:.2f} ref={refined.score:.2f}/"
                f"{refined.template_score:.2f}/{refined.layout_score:.2f}/"
                f"{refined.strip_score:.2f}/{refined.border_score:.2f}/"
                f"{refined.led_center_score:.2f}/{refined.square_anchor_score:.2f}/{refined.triangle_anchor_score:.2f}"
            ),
        )

    def _try_soft_acquire(self, frame_bgr: np.ndarray, gray: np.ndarray) -> TrackerResult | None:
        candidate = self.detector.detect(frame_bgr, None, True)
        if candidate is None:
            return None
        refined = refine_pose(frame_bgr, candidate.pose)
        strong_strip = refined.strip_score >= 0.20
        strong_border_layout = refined.border_score >= 0.20 and refined.layout_score >= 0.70
        layout_acquire_ok = (
            refined.score >= 0.50
            and refined.template_score >= 0.86
            and refined.layout_score >= 0.60
            and (strong_strip or strong_border_layout)
            and refined.border_score >= 0.12
            and refined.led_center_score >= 0.55
            and refined.square_anchor_score >= 0.88
            and refined.triangle_anchor_score >= 0.84
        )
        anchored_acquire_ok = (
            refined.score >= 0.50
            and refined.template_score >= 0.86
            and refined.layout_score >= 0.40
            and refined.strip_score >= 0.40
            and refined.border_score >= 0.24
            and refined.led_center_score >= 0.70
            and refined.square_anchor_score >= 0.90
            and refined.triangle_anchor_score >= 0.90
        )
        acquire_ok = layout_acquire_ok or anchored_acquire_ok
        if not acquire_ok:
            return None
        self.was_tracking = True
        self._accept_pose(refined.pose, gray)
        self.missed_frames = 0
        self.established_frames = 1
        return self._pose_result(
            frame_bgr,
            refined.pose,
            "acquire_soft",
            candidate.score,
            (
                f"cv_soft_acquire sq={candidate.square_score:.2f} tri={candidate.triangle_score:.2f} "
                f"geom={candidate.geometry_score:.2f} tmpl={candidate.template_score:.2f} "
                f"layout={candidate.layout_score:.2f} strip={candidate.strip_score:.2f} "
                f"prior={candidate.prior_score:.2f} led={candidate.led_score:.2f} "
                f"nn={candidate.nn_score:.2f} ref={refined.score:.2f}/"
                f"{refined.template_score:.2f}/{refined.layout_score:.2f}/"
                f"{refined.strip_score:.2f}/{refined.border_score:.2f}/"
                f"{refined.led_center_score:.2f}/{refined.square_anchor_score:.2f}/{refined.triangle_anchor_score:.2f}"
            ),
        )

    def _flow_pose(self, gray: np.ndarray) -> Pose | None:
        if self.flow_gray is None or self.flow_pose is None or self.missed_frames >= self.max_tracking_misses:
            return None
        source_points = np.array([self.flow_pose.square, *self.flow_pose.leds, self.flow_pose.triangle], dtype=np.float32).reshape(-1, 1, 2)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.flow_gray,
            gray,
            source_points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 18, 0.02),
        )
        if next_points is None or status is None:
            return None
        valid = status.reshape(-1).astype(bool)
        if int(valid.sum()) < 4:
            return None
        src = source_points.reshape(-1, 2)[valid]
        dst = next_points.reshape(-1, 2)[valid]
        transform, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=60)
        if transform is None or inliers is None or int(inliers.sum()) < 4:
            return None
        center = np.array([[self.flow_pose.center]], dtype=np.float32)
        square = np.array([[self.flow_pose.square]], dtype=np.float32)
        triangle = np.array([[self.flow_pose.triangle]], dtype=np.float32)
        center_new = cv2.transform(center, transform).reshape(2)
        square_new = cv2.transform(square, transform).reshape(2)
        triangle_new = cv2.transform(triangle, transform).reshape(2)
        dx = float(triangle_new[0] - square_new[0])
        dy = float(triangle_new[1] - square_new[1])
        scale = math.hypot(dx, dy)
        if scale < 8.0:
            return None
        return Pose(center=(float(center_new[0]), float(center_new[1])), angle_rad=math.atan2(dy, dx), scale_px=scale)

    def _predicted_pose(self) -> Pose | None:
        if self.last_real_pose is None:
            return self.last_pose
        dt = max(0, self.frame_index - self.last_real_frame)
        dt = min(dt, self.max_tracking_misses + 1)
        if dt <= 1:
            effective_dt = float(dt)
        else:
            # A missed frame is not a measurement. Keep one-frame prediction sharp,
            # then quickly saturate extrapolation so a bad velocity estimate cannot
            # walk the search window away from the real marker.
            decay = 0.58
            effective_dt = float((1.0 - decay**dt) / (1.0 - decay))
        scale = self.last_real_pose.scale_px * math.exp(self.velocity_log_scale * effective_dt)
        return Pose(
            center=(
                self.last_real_pose.center[0] + self.velocity_x * effective_dt,
                self.last_real_pose.center[1] + self.velocity_y * effective_dt,
            ),
            angle_rad=self.last_real_pose.angle_rad + self.velocity_angle * effective_dt,
            scale_px=scale,
        )

    def _pose_result(self, frame_bgr: np.ndarray, pose: Pose, mode: str, confidence: float, debug: str) -> TrackerResult:
        return TrackerResult(
            hit=True,
            mode=mode,
            confidence=confidence,
            x=pose.center[0],
            y=pose.center[1],
            angle_rad=pose.angle_rad,
            scale_px=pose.scale_px,
            square=pose.square,
            triangle=pose.triangle,
            leds=pose.leds,
            led_scores=led_scores_from_pose(frame_bgr, pose),
            debug=debug,
        )

    def _hold_result(self, frame_bgr: np.ndarray, pose: Pose | None, debug: str) -> TrackerResult:
        if pose is None:
            h, w = frame_bgr.shape[:2]
            return TrackerResult(hit=False, mode="acquire", x=w * 0.5, y=h * 0.5, debug=debug)
        return TrackerResult(
            hit=False,
            mode="track_predict",
            confidence=0.0,
            x=pose.center[0],
            y=pose.center[1],
            angle_rad=pose.angle_rad,
            scale_px=pose.scale_px,
            square=pose.square,
            triangle=pose.triangle,
            leds=pose.leds,
            led_scores=[],
            debug=debug,
        )
