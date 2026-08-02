"""MARS and development baselines for RFI binary segmentation.

The goal is deliberately narrow: preserve full-resolution segmentation while
using small channel counts and deployment-friendly operations.

Four variants are exposed:

``LightUNet512V1``
    The first validation model.  It keeps additive skip connections and adds an
    axis-context bottleneck for long horizontal/vertical RFI structures.

``LightUNet512V2``
    A simpler follow-up model.  It keeps the same full-resolution decoder but
    replaces the axis-context bottleneck with a local anisotropic context block.

``TRTFastUNet512``
    A TensorRT-friendly validation model.  It uses standard Conv-BN-ReLU blocks,
    additive skips, and a small dilated bottleneck instead of depthwise/group-norm
    blocks.  It has more parameters than V1, but far fewer awkward kernels for
    TensorRT to schedule.

``TRTShapeUNet512``
    The MARS architecture described in the paper. It combines the
    TensorRT-friendly encoder/decoder with local, horizontal, and vertical
    bottleneck branches and horizontal/vertical refinement at all three decoder
    scales. Its paper defaults contain 270,769 trainable parameters.

The other classes are retained as explicit development baselines; they are not
the model reported as MARS in the paper.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 + GroupNorm + SiLU."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                stride=int(stride),
                padding=1,
                groups=in_ch,
                bias=False,
            ),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DSBlock(nn.Module):
    """Residual depthwise-separable block, optionally downsampling."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch, stride=stride)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch, stride=1)
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=int(stride), bias=False),
                nn.GroupNorm(_num_groups(out_ch), out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + self.skip(x)


class AxisContextBlock(nn.Module):
    """Frequency/time axis gates plus light residual refinement.

    This is the V1-specific context block.  It gives the bottleneck a cheap way
    to reason about whole rows and columns without introducing a large semantic
    segmentation backbone.
    """

    def __init__(self, channels: int, *, reduction: int = 4):
        super().__init__()
        hidden = max(1, int(channels) // int(reduction))
        self.freq_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.time_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.global_bias = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.refine = DSBlock(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = x.mean(dim=3, keepdim=True)
        time = x.mean(dim=2, keepdim=True)
        gated = x * self.freq_gate(freq) * self.time_gate(time)
        gated = gated + self.global_bias(x)
        return self.refine(gated) + x


class AnisotropicContextBlock(nn.Module):
    """Local horizontal/vertical context block for the simpler V2 model."""

    def __init__(self, channels: int, *, kernel_size: int = 9):
        super().__init__()
        if int(kernel_size) % 2 != 1:
            raise ValueError("kernel_size must be odd")
        k = int(kernel_size)
        self.horizontal = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, k),
            padding=(0, k // 2),
            groups=channels,
            bias=False,
        )
        self.vertical = nn.Conv2d(
            channels,
            channels,
            kernel_size=(k, 1),
            padding=(k // 2, 0),
            groups=channels,
            bias=False,
        )
        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.SiLU(inplace=True),
        )
        self.refine = DSBlock(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.horizontal(x), self.vertical(x), self.local(x)], dim=1)
        return self.refine(self.fuse(y)) + x


class UpAddBlock(nn.Module):
    """Bilinear upsample, additive skip fusion, and light refinement."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up_proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
        )
        if int(skip_ch) == int(out_ch):
            self.skip_proj = nn.Identity()
        else:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False),
                nn.GroupNorm(_num_groups(out_ch), out_ch),
            )
        self.refine = DSBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(self.up_proj(x) + self.skip_proj(skip))


class ConvBNAct(nn.Module):
    """TensorRT-friendly Conv2d + BatchNorm2d + ReLU block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int | tuple[int, int] = 3,
        stride: int = 1,
        dilation: int | tuple[int, int] = 1,
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kh, kw = int(kernel_size[0]), int(kernel_size[1])
        else:
            kh = kw = int(kernel_size)
        if isinstance(dilation, tuple):
            dh, dw = int(dilation[0]), int(dilation[1])
        else:
            dh = dw = int(dilation)
        padding = (((kh - 1) // 2) * dh, ((kw - 1) // 2) * dw)
        self.net = nn.Sequential(
            nn.Conv2d(
                int(in_ch),
                int(out_ch),
                kernel_size=(kh, kw),
                stride=int(stride),
                padding=padding,
                dilation=(dh, dw),
                bias=False,
            ),
            nn.BatchNorm2d(int(out_ch)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FastUpAddBlock(nn.Module):
    """Nearest upsample + additive skip fusion + one Conv-BN-ReLU refinement."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up_proj = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, bias=False)
        if int(skip_ch) == int(out_ch):
            self.skip_proj = nn.Identity()
        else:
            self.skip_proj = nn.Conv2d(int(skip_ch), int(out_ch), kernel_size=1, bias=False)
        self.refine = ConvBNAct(int(out_ch), int(out_ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.refine(self.up_proj(x) + self.skip_proj(skip))


class RFIShapeContextBlock(nn.Module):
    """Shape-aware bottleneck for horizontal, vertical, and compact RFI."""

    def __init__(self, channels: int, *, kernel_size: int = 9):
        super().__init__()
        if int(kernel_size) % 2 != 1:
            raise ValueError("shape_kernel must be odd")
        k = int(kernel_size)
        self.local = ConvBNAct(channels, channels, kernel_size=3)
        self.horizontal = ConvBNAct(channels, channels, kernel_size=(1, k))
        self.vertical = ConvBNAct(channels, channels, kernel_size=(k, 1))
        self.fuse = ConvBNAct(3 * channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat(
            [
                self.local(x),
                self.horizontal(x),
                self.vertical(x),
            ],
            dim=1,
        )
        return self.fuse(y) + x


class HorizontalRefineBlock(nn.Module):
    """High-resolution horizontal refinement for long bright RFI streaks."""

    def __init__(self, channels: int, *, kernel_size: int = 15):
        super().__init__()
        if int(kernel_size) % 2 != 1:
            raise ValueError("decoder_horizontal_refine_kernel must be odd")
        k = int(kernel_size)
        self.net = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=(1, k)),
            ConvBNAct(channels, channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class VerticalRefineBlock(nn.Module):
    """High-resolution vertical refinement for short broadband RFI bursts."""

    def __init__(self, channels: int, *, kernel_size: int = 9):
        super().__init__()
        if int(kernel_size) % 2 != 1:
            raise ValueError("decoder_vertical_refine_kernel must be odd")
        k = int(kernel_size)
        self.net = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=(k, 1)),
            ConvBNAct(channels, channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


def _normalize_decoder_refine_stages(stages, *, name: str = "decoder_refine_stages") -> set[str]:
    if stages is None:
        return set()
    if isinstance(stages, str):
        text = stages.strip()
        if not text or text.lower() in ("none", "off", "false", "0"):
            return set()
        parts = [p.strip().lower() for p in text.split(",")]
    else:
        parts = [str(p).strip().lower() for p in stages]
    out = {p for p in parts if p}
    allowed = {"up2", "up1", "up0"}
    invalid = sorted(out - allowed)
    if invalid:
        raise ValueError(
            f"{name} must contain only "
            f"{sorted(allowed)}; got invalid stages {invalid}"
        )
    return out


class TRTFastUNet512(nn.Module):
    """TensorRT-friendly full-resolution UNet for RFI segmentation.

    This model intentionally avoids depthwise convolutions, GroupNorm, axis
    reductions, and sigmoid gates.  The bottleneck uses standard 3x3 convolutions
    plus one optional dilated convolution to keep some long-structure context
    while remaining easy for TensorRT to optimize.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] | Iterable[int] = (8, 16, 24, 32),
        output_bias_prior: float = 0.0,
        bottleneck_dilation: int = 2,
        decoder_horizontal_refine_enabled: bool = False,
        decoder_horizontal_refine_stages: Sequence[str] | str = ("up1",),
        decoder_horizontal_refine_kernel: int = 15,
        decoder_vertical_refine_enabled: bool = False,
        decoder_vertical_refine_stages: Sequence[str] | str = (),
        decoder_vertical_refine_kernel: int = 9,
    ):
        super().__init__()
        ch = tuple(int(c) for c in channels)
        if len(ch) != 4:
            raise ValueError("TRTFastUNet512 expects exactly four channel widths")
        self.channels = ch

        self.stem = ConvBNAct(int(in_channels), ch[0])
        self.down1 = ConvBNAct(ch[0], ch[1], stride=2)
        self.down2 = ConvBNAct(ch[1], ch[2], stride=2)
        self.down3 = ConvBNAct(ch[2], ch[3], stride=2)

        self.bottleneck = nn.Sequential(
            ConvBNAct(ch[3], ch[3]),
            ConvBNAct(ch[3], ch[3], dilation=max(1, int(bottleneck_dilation))),
        )

        self.up2 = FastUpAddBlock(ch[3], ch[2], ch[2])
        self.up1 = FastUpAddBlock(ch[2], ch[1], ch[1])
        self.up0 = FastUpAddBlock(ch[1], ch[0], ch[0])
        horizontal_refine_stages = (
            _normalize_decoder_refine_stages(
                decoder_horizontal_refine_stages,
                name="decoder_horizontal_refine_stages",
            )
            if bool(decoder_horizontal_refine_enabled)
            else set()
        )
        vertical_refine_stages = (
            _normalize_decoder_refine_stages(
                decoder_vertical_refine_stages,
                name="decoder_vertical_refine_stages",
            )
            if bool(decoder_vertical_refine_enabled)
            else set()
        )
        horizontal_refine_kernel = int(decoder_horizontal_refine_kernel)
        vertical_refine_kernel = int(decoder_vertical_refine_kernel)
        self.refine_up2 = (
            HorizontalRefineBlock(ch[2], kernel_size=horizontal_refine_kernel)
            if "up2" in horizontal_refine_stages else nn.Identity()
        )
        self.refine_up1 = (
            HorizontalRefineBlock(ch[1], kernel_size=horizontal_refine_kernel)
            if "up1" in horizontal_refine_stages else nn.Identity()
        )
        self.refine_up0 = (
            HorizontalRefineBlock(ch[0], kernel_size=horizontal_refine_kernel)
            if "up0" in horizontal_refine_stages else nn.Identity()
        )
        self.vertical_refine_up2 = (
            VerticalRefineBlock(ch[2], kernel_size=vertical_refine_kernel)
            if "up2" in vertical_refine_stages else nn.Identity()
        )
        self.vertical_refine_up1 = (
            VerticalRefineBlock(ch[1], kernel_size=vertical_refine_kernel)
            if "up1" in vertical_refine_stages else nn.Identity()
        )
        self.vertical_refine_up0 = (
            VerticalRefineBlock(ch[0], kernel_size=vertical_refine_kernel)
            if "up0" in vertical_refine_stages else nn.Identity()
        )
        self.head = nn.Conv2d(ch[0], int(out_channels), kernel_size=1)
        nn.init.constant_(self.head.bias, float(output_bias_prior))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        h = self.down3(s2)
        h = self.bottleneck(h)
        h = self.up2(h, s2)
        h = self.refine_up2(h)
        h = self.vertical_refine_up2(h)
        h = self.up1(h, s1)
        h = self.refine_up1(h)
        h = self.vertical_refine_up1(h)
        h = self.up0(h, s0)
        h = self.refine_up0(h)
        h = self.vertical_refine_up0(h)
        return self.head(h)


class TRTShapeUNet512(TRTFastUNet512):
    """Paper MARS model with a morphology-aware bottleneck and decoder."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] | Iterable[int] = (8, 16, 32, 64),
        output_bias_prior: float = 0.0,
        shape_kernel: int = 9,
        decoder_horizontal_refine_enabled: bool = True,
        decoder_horizontal_refine_stages: Sequence[str] | str = ("up2", "up1", "up0"),
        decoder_horizontal_refine_kernel: int = 31,
        decoder_vertical_refine_enabled: bool = True,
        decoder_vertical_refine_stages: Sequence[str] | str = ("up2", "up1", "up0"),
        decoder_vertical_refine_kernel: int = 31,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            output_bias_prior=output_bias_prior,
            bottleneck_dilation=1,
            decoder_horizontal_refine_enabled=decoder_horizontal_refine_enabled,
            decoder_horizontal_refine_stages=decoder_horizontal_refine_stages,
            decoder_horizontal_refine_kernel=decoder_horizontal_refine_kernel,
            decoder_vertical_refine_enabled=decoder_vertical_refine_enabled,
            decoder_vertical_refine_stages=decoder_vertical_refine_stages,
            decoder_vertical_refine_kernel=decoder_vertical_refine_kernel,
        )
        self.bottleneck = RFIShapeContextBlock(self.channels[-1], kernel_size=int(shape_kernel))


class _BaseLightUNet512(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] | Iterable[int] = (8, 16, 24, 32),
        output_bias_prior: float = 0.0,
    ):
        super().__init__()
        ch = tuple(int(c) for c in channels)
        if len(ch) != 4:
            raise ValueError("LightUNet512 expects exactly four channel widths")
        self.channels = ch

        self.stem = DSBlock(int(in_channels), ch[0], stride=1)
        self.down1 = DSBlock(ch[0], ch[1], stride=2)
        self.down2 = DSBlock(ch[1], ch[2], stride=2)
        self.down3 = DSBlock(ch[2], ch[3], stride=2)

        self.up2 = UpAddBlock(ch[3], ch[2], ch[2])
        self.up1 = UpAddBlock(ch[2], ch[1], ch[1])
        self.up0 = UpAddBlock(ch[1], ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], int(out_channels), kernel_size=1)
        nn.init.constant_(self.head.bias, float(output_bias_prior))

    def _context(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        h = self.down3(s2)
        h = self._context(h)
        h = self.up2(h, s2)
        h = self.up1(h, s1)
        h = self.up0(h, s0)
        return self.head(h)


class LightUNet512V1(_BaseLightUNet512):
    """First validation model: light full-resolution UNet with axis context."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] | Iterable[int] = (8, 16, 24, 32),
        output_bias_prior: float = 0.0,
        axis_reduction: int = 4,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            output_bias_prior=output_bias_prior,
        )
        self.context = AxisContextBlock(self.channels[-1], reduction=int(axis_reduction))

    def _context(self, x: torch.Tensor) -> torch.Tensor:
        return self.context(x)


class LightUNet512V2(_BaseLightUNet512):
    """Simpler follow-up model: same decoder, local anisotropic context only."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] | Iterable[int] = (8, 16, 24, 32),
        output_bias_prior: float = 0.0,
        anisotropic_kernel: int = 9,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            output_bias_prior=output_bias_prior,
        )
        self.context = AnisotropicContextBlock(self.channels[-1], kernel_size=int(anisotropic_kernel))

    def _context(self, x: torch.Tensor) -> torch.Tensor:
        return self.context(x)


MODEL_ALIASES = {
    "light_unet_v1": LightUNet512V1,
    "lightunetv1": LightUNet512V1,
    "light_unet512_v1": LightUNet512V1,
    "lightunet512v1": LightUNet512V1,
    "v1": LightUNet512V1,
    "light_unet_v2": LightUNet512V2,
    "lightunetv2": LightUNet512V2,
    "light_unet512_v2": LightUNet512V2,
    "lightunet512v2": LightUNet512V2,
    "v2": LightUNet512V2,
    "trt_fast_unet": TRTFastUNet512,
    "trt_fast_unet512": TRTFastUNet512,
    "trtfastunet": TRTFastUNet512,
    "trtfastunet512": TRTFastUNet512,
    "fast_unet": TRTFastUNet512,
    "fast_unet512": TRTFastUNet512,
    "fast": TRTFastUNet512,
    "trt_shape_unet": TRTShapeUNet512,
    "trt_shape_unet512": TRTShapeUNet512,
    "trtshapeunet": TRTShapeUNet512,
    "trtshapeunet512": TRTShapeUNet512,
    "shape_unet": TRTShapeUNet512,
    "shape_unet512": TRTShapeUNet512,
    "shape": TRTShapeUNet512,
}


def _cfg_get(config: Mapping, key: str, default):
    if key in config:
        return config[key]
    model_cfg = config.get("model_config", {})
    if isinstance(model_cfg, Mapping) and key in model_cfg:
        return model_cfg[key]
    return default


def build_model(config: Mapping | None = None, **overrides) -> nn.Module:
    """Build the configured MARS model or an explicit development baseline.

    ``config`` can be the full training config saved in a checkpoint.  The model
    choice is controlled by the top-level ``model`` key:

    - ``"light_unet_v1"`` for the axis-context model.
    - ``"light_unet_v2"`` for the simpler anisotropic-context model.
    - ``"trt_fast_unet"`` for the TensorRT-friendly Conv-BN-ReLU model.
    - ``"trt_shape_unet"`` for the RFI shape-aware TensorRT-friendly model.
    """

    cfg: dict = {}
    if config is not None:
        cfg.update(dict(config))
    cfg.update(overrides)

    name = str(_cfg_get(cfg, "model", "trt_shape_unet")).lower()
    try:
        cls = MODEL_ALIASES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model {name!r}. Expected one of: {', '.join(sorted(MODEL_ALIASES))}"
        ) from exc

    default_channels = (8, 16, 32, 64) if cls is TRTShapeUNet512 else (8, 16, 24, 32)
    channels = tuple(_cfg_get(cfg, "channels", default_channels))
    kwargs = {
        "in_channels": int(_cfg_get(cfg, "in_channels", 1)),
        "out_channels": int(_cfg_get(cfg, "out_channels", 1)),
        "channels": channels,
        "output_bias_prior": float(_cfg_get(cfg, "output_bias_prior", 0.0)),
    }
    if cls is LightUNet512V1:
        kwargs["axis_reduction"] = int(_cfg_get(cfg, "axis_reduction", 4))
    if cls is LightUNet512V2:
        kwargs["anisotropic_kernel"] = int(_cfg_get(cfg, "anisotropic_kernel", 9))
    if cls in (TRTFastUNet512, TRTShapeUNet512):
        paper_shape_model = cls is TRTShapeUNet512
        kwargs["decoder_horizontal_refine_enabled"] = bool(
            _cfg_get(cfg, "decoder_horizontal_refine_enabled", paper_shape_model)
        )
        kwargs["decoder_horizontal_refine_stages"] = _cfg_get(
            cfg,
            "decoder_horizontal_refine_stages",
            ("up2", "up1", "up0") if paper_shape_model else ("up1",),
        )
        kwargs["decoder_horizontal_refine_kernel"] = int(
            _cfg_get(cfg, "decoder_horizontal_refine_kernel", 31 if paper_shape_model else 15)
        )
        kwargs["decoder_vertical_refine_enabled"] = bool(
            _cfg_get(cfg, "decoder_vertical_refine_enabled", paper_shape_model)
        )
        kwargs["decoder_vertical_refine_stages"] = _cfg_get(
            cfg,
            "decoder_vertical_refine_stages",
            ("up2", "up1", "up0") if paper_shape_model else (),
        )
        kwargs["decoder_vertical_refine_kernel"] = int(
            _cfg_get(cfg, "decoder_vertical_refine_kernel", 31 if paper_shape_model else 9)
        )
    if cls is TRTFastUNet512:
        kwargs["bottleneck_dilation"] = int(_cfg_get(cfg, "bottleneck_dilation", 2))
    if cls is TRTShapeUNet512:
        kwargs["shape_kernel"] = int(_cfg_get(cfg, "shape_kernel", 9))
    return cls(**kwargs)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


PAPER_PARAMETER_COUNT = 270_769


def build_paper_model(**overrides) -> TRTShapeUNet512:
    """Build the architecture reported in the MARS paper.

    Overrides make controlled ablations possible. Callers should set or remove
    their own parameter-count guard when changing the architecture.
    """

    return TRTShapeUNet512(**overrides)
