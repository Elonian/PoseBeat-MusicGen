from __future__ import annotations

from diffusers import UNet2DConditionModel


def create_motion_conditioned_unet(
    *,
    sample_size: tuple[int, int],
    cross_attention_dim: int,
    in_channels: int = 1,
    out_channels: int = 1,
    variant: str = "base",
) -> UNet2DConditionModel:
    variant = variant.lower()
    if variant in {"base", "motion_to_music_256"}:
        return UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=cross_attention_dim,
        )

    if variant in {"wide_64", "motion_to_music_hires"}:
        return UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512, 512),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=8,
        )

    raise ValueError(f"unsupported motion-conditioned UNet variant: {variant}")
