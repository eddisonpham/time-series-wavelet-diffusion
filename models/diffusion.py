from diffusers import UNet2DModel, DDPMScheduler
from config import IMAGE_SIZE, NUM_TIMESTEPS


def create_model():
    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
    return model, scheduler