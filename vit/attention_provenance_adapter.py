"""
Captures ViT patch-attention scores as Flowcept provenance via the "timm"
adapter (flowcept/src/flowcept/flowceptor/adapters/timm/): a native
torch.nn.Module.register_forward_hook, attached once, that transparently
emits a task on every subsequent forward call.

`run_workflow()` below has zero Flowcept API in its body -- no FlowceptTask,
FlowceptLoop, or @flowcept_task, only `@flowcept` on the function itself plus
one `TimmInterceptor.attach()` call up front. Trade-off: since nothing tracks
it, Dice is printed but not captured as provenance -- inherent to the
adapter pattern, which only observes what it's attached to, exactly like the
Dask adapter only ever sees `client.submit(...)` calls.

`register_forward_hook` only fires on `__call__`/`forward()`, so this calls
`native(x)` rather than `native.forward_features(x)`.

appl-rgb-segmentation is unmodified. No real APPL imagery or trained
checkpoint exists here, so this uses synthetic images and ImageNet-pretrained
(not fine-tuned) weights, purely to verify the provenance trace end to end.

Run: python examples/vit/attention_provenance_adapter.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
APPL_DIR = REPO_ROOT / "appl-rgb-segmentation"
FLOWCEPT_SRC = REPO_ROOT / "flowcept" / "src"
TIMM_FORK = REPO_ROOT / "pytorch-image-models"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
BUFFER_PATH = OUTPUT_DIR / "flowcept_buffer_adapter_example.jsonl"

sys.path.insert(0, str(FLOWCEPT_SRC))  # local edited clone -- has TimmInterceptor
sys.path.insert(0, str(TIMM_FORK))     # real GitHub fork -- has the attention-capture instrumentation
sys.path.insert(0, str(APPL_DIR))
import test as appl_test      # noqa: E402
import models as appl_models  # noqa: E402

import flowcept  # noqa: E402
from flowcept import Flowcept  # noqa: E402
from flowcept.instrumentation.flowcept_decorator import flowcept as flowcept_workflow  # noqa: E402
from flowcept.flowceptor.adapters.timm.timm_interceptor import TimmInterceptor  # noqa: E402

import timm  # noqa: E402
assert str(TIMM_FORK) in timm.__file__, timm.__file__
assert str(FLOWCEPT_SRC) in flowcept.__file__, flowcept.__file__

TILE_SIZE = 224
N_IMAGES = 3


def get_native_vit(model: torch.nn.Module):
    """Unwrap torch.compile + FeatureGetterNet down to the native VisionTransformer."""
    m = getattr(model, "_orig_mod", model)
    return m.encoder.model


def prepare_center_tile(image: np.ndarray, tile_size: int, device) -> torch.Tensor:
    """Crop+normalize a center tile, ready for a plain `native(x)` call."""
    H, W, _ = image.shape
    ws = min(tile_size, H, W)
    r, c = (H - ws) // 2, (W - ws) // 2
    tile = appl_test._normalize_image(image[r:r + ws, c:c + ws, :])
    return torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).float().to(device)


def make_synthetic_image(seed: int, size: int = 512):
    """Stand-in for a real loaded image: uint8 HxWx3 with a foreground blob."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx, r = size * 0.5, size * (0.3 + 0.1 * seed), size * 0.2
    blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    img = np.full((size, size, 3), 40, dtype=np.uint8)
    img[blob] = [30, 160, 40]
    noise = rng.normal(0, 5, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8), blob


@flowcept_workflow(
    interceptors="timm",  # the only interceptor needed; nothing here uses FlowceptTask/FlowceptLoop
    workflow_name="vit_attention_provenance_adapter",
    workflow_args={"backbone": "vit_small_patch16_224", "tile_size": TILE_SIZE},
    save_workflow=True,  # @flowcept defaults this to False, unlike plain Flowcept()
)
def run_workflow(model, native, device):
    """Plain business logic -- no Flowcept API anywhere in this body."""
    for i in range(N_IMAGES):
        image, blob_mask = make_synthetic_image(seed=i)
        preds, _ = appl_test.segment(
            image, model, tile_size=TILE_SIZE, batch_size=8,
            n_classes=2, inference_mode="hann")

        x = prepare_center_tile(image, TILE_SIZE, device)
        with torch.no_grad():
            native(x)  # plain call -- TimmInterceptor's hook captures it automatically

        dice = appl_test.compute_image_dice(preds, blob_mask.astype(np.uint8))
        print(f"image {i}: dice={dice:.4f} (printed only, not provenance)")

    Flowcept.get_current_instance().dump_buffer(str(BUFFER_PATH))
    return Flowcept.current_workflow_id


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    if BUFFER_PATH.exists():
        BUFFER_PATH.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = appl_models.build_model(
        "vit", "vit_small_patch16_224", tile_size=TILE_SIZE, device=device, pretrained=True)
    model.eval()
    native = get_native_vit(model)

    TimmInterceptor.get_instance().attach(native, activity_id="vit_forward")
    workflow_id = run_workflow(model, native, device)
    TimmInterceptor.get_instance().detach_all()

    print(f"\nworkflow_id = {workflow_id}")
    print(f"buffer dumped to {BUFFER_PATH} ({BUFFER_PATH.stat().st_size} bytes)")

    records = Flowcept.read_buffer_file(str(BUFFER_PATH))
    workflows = [r for r in records if r.get("type") == "workflow"]
    attn_tasks = [r for r in records if r.get("activity_id") == "vit_forward"]

    print(f"\ncaptured {len(records)} total records: "
          f"{len(workflows)} workflow(s), {len(attn_tasks)} vit_forward task(s)")

    assert len(records) == 1 + N_IMAGES
    assert len(workflows) == 1 and workflows[0]["workflow_id"] == workflow_id
    assert len(attn_tasks) == N_IMAGES
    for t in attn_tasks:
        assert t["workflow_id"] == workflow_id
        scores = t["generated"]["patch_attention"]
        assert isinstance(scores, list) and len(scores) == (TILE_SIZE // 16) ** 2
        assert all(isinstance(v, float) for v in scores)
    print("\nALL CHECKS PASSED -- attention scores captured transparently via the timm adapter.")


if __name__ == "__main__":
    main()
