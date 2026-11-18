
"""
Model worker for Hunyuan3D API server.

RTX 4090 optimized version:
- Lazy load Shape / Paint pipelines.
- Release Shape pipeline before loading Paint pipeline.
- Serialize GPU generation tasks.
- Preserve original API and generation workflow.
"""
import os
import gc
import time
import uuid
import base64
import threading
import trimesh

from io import BytesIO
from PIL import Image

import torch

# Apply torchvision compatibility fix before other imports
import sys

sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print(
        "Warning: torchvision_fix module not found, "
        "proceeding without compatibility fix"
    )
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.utils import logger

from textureGenPipeline import (
    Hunyuan3DPaintPipeline,
    Hunyuan3DPaintConfig
)

from hy3dpaint.convert_utils import create_glb_with_pbr_materials


def cuda_cleanup():
    """
    Release unused CUDA memory.
    """
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def quick_convert_with_obj2gltf(obj_path: str, glb_path: str):
    """
    Preserve original PBR conversion implementation.
    """
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),
        'roughness': obj_path.replace('.obj', '_roughness.jpg')
    }

    create_glb_with_pbr_materials(
        obj_path,
        textures,
        glb_path
    )


def load_image_from_base64(image):
    """
    Load image from base64 encoded string.
    """
    if isinstance(image, Image.Image):
        return image.copy()

    if not isinstance(image, str):
        raise ValueError("Image must be a base64 encoded string")

    # Support data URI:
    # data:image/png;base64,xxxx
    if ',' in image and image.startswith('data:'):
        image = image.split(',', 1)[1]

    image_bytes = base64.b64decode(image)

    return Image.open(BytesIO(image_bytes)).copy()


class ModelWorker:
    """
    Worker class for Hunyuan3D model generation.

    Shape and Paint pipelines are loaded on demand.
    Only one GPU generation task is executed at a time.
    """

    def __init__(
        self,
        model_path='tencent/Hunyuan3D-2.1',
        subfolder='hunyuan3d-dit-v2-1',
        device='cuda',
        low_vram_mode=False,
        worker_id=None,
        model_semaphore=None,
        save_dir='gradio_cache',
        mc_algo='mc',
        enable_flashvdm=False,
        compile=False
    ):
        self.model_path = model_path
        self.subfolder = subfolder
        self.worker_id = worker_id or str(uuid.uuid4())[:6]
        self.device = device
        self.low_vram_mode = low_vram_mode
        self.model_semaphore = model_semaphore
        self.save_dir = save_dir
        self.mc_algo = mc_algo
        self.enable_flashvdm = enable_flashvdm
        self.compile = compile

        # Pipeline references.
        self.pipeline = None
        self.paint_pipeline = None

        # Worker-level GPU lock.
        self.generation_lock = threading.RLock()

        # Background remover.
        self.rembg = None

        os.makedirs(self.save_dir, exist_ok=True)

        logger.info(
            f"Initializing ModelWorker {self.worker_id}, "
            f"model={model_path}, device={device}"
        )

        # Background remover is kept separate from Shape/Paint.
        self.rembg = BackgroundRemover()

        # Do not initialize both pipelines here.
        # They will be loaded on demand.

        # Clean old cache files.
        self._clean_save_dir()

        logger.info(
            f"ModelWorker {self.worker_id} initialized. "
            "Shape/Paint pipelines will be loaded lazily."
        )

    def _clean_save_dir(self):
        """
        Clean old generated files.

        Only files are removed. Directories are preserved.
        """
        if not os.path.isdir(self.save_dir):
            return

        for filename in os.listdir(self.save_dir):
            path = os.path.join(self.save_dir, filename)

            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(
                    f"Could not remove old cache file {path}: {e}"
                )

    def _load_shape_pipeline(self):
        """
        Load Shape pipeline on demand.
        """
        if self.pipeline is not None:
            return self.pipeline

        logger.info("Loading Shape pipeline...")

        cuda_cleanup()

        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.model_path
        )

        if self.enable_flashvdm:
            mc_algo = (
                'mc'
                if self.device in ['cpu', 'mps']
                else self.mc_algo
            )

            pipeline.enable_flashvdm(mc_algo=mc_algo)

        if self.compile:
            pipeline.compile()

        self.pipeline = pipeline

        logger.info("Shape pipeline loaded.")

        return self.pipeline

    def _load_paint_pipeline(self):
        """
        Load Paint pipeline on demand.

        Shape pipeline must be released before this method is called.
        """
        if self.paint_pipeline is not None:
            return self.paint_pipeline

        if self.pipeline is not None:
            raise RuntimeError(
                "Cannot load Paint pipeline while Shape pipeline "
                "is still referenced."
            )

        logger.info("Loading Paint pipeline...")

        cuda_cleanup()

        max_num_view = 6
        resolution = 512

        conf = Hunyuan3DPaintConfig(
            max_num_view,
            resolution
        )

        conf.realesrgan_ckpt_path = (
            "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        )

        conf.multiview_cfg_path = (
            "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        )

        conf.custom_pipeline = (
            "hy3dpaint/hunyuanpaintpbr"
        )

        self.paint_pipeline = Hunyuan3DPaintPipeline(conf)

        logger.info("Paint pipeline loaded.")

        return self.paint_pipeline

    def _release_shape_pipeline(self):
        """
        Release Shape pipeline completely.
        """
        if self.pipeline is not None:
            logger.info("Releasing Shape pipeline...")

            try:
                del self.pipeline
            finally:
                self.pipeline = None

        cuda_cleanup()

        logger.info("Shape pipeline released.")

    def _release_paint_pipeline(self):
        """
        Release Paint pipeline completely.
        """
        if self.paint_pipeline is not None:
            logger.info("Releasing Paint pipeline...")

            try:
                del self.paint_pipeline
            finally:
                self.paint_pipeline = None

        cuda_cleanup()

        logger.info("Paint pipeline released.")

    def get_queue_length(self):
        """
        Return queue length.
        """
        if self.model_semaphore is None:
            return 0

        return (
            self.model_semaphore._value
            if hasattr(self.model_semaphore, '_value')
            else 0
        ) + (
            len(self.model_semaphore._waiters)
            if hasattr(self.model_semaphore, '_waiters')
            and self.model_semaphore._waiters is not None
            else 0
        )

    def get_status(self):
        """
        Return worker status.
        """
        return {
            "speed": 1,
            "queue_length": self.get_queue_length()
        }

    def _prepare_image(self, image):
        """
        Prepare input image.

        Keep the original RGB image for background removal,
        then convert to RGBA for the generation pipeline.
        """
        image = image.convert("RGB")

        # Preserve the original behavior intention:
        # remove background when RGB input is provided.
        if self.rembg is not None:
            image = self.rembg(image)

        return image.convert("RGBA")

    @torch.inference_mode()
    def generate(self, uid, params):
        """
        Generate a 3D model.

        Original interface preserved:
            generate(uid, params)

        Returns:
            (file_path, uid)
        """
        with self.generation_lock:
            return self._generate_locked(uid, params)

    def _generate_locked(self, uid, params):
        start_time = time.time()

        logger.info(
            f"Generating 3D model for uid: {uid}"
        )

        if 'image' not in params:
            raise ValueError("No input image provided")

        image = load_image_from_base64(
            params["image"]
        )

        image = self._prepare_image(image)

        os.makedirs(self.save_dir, exist_ok=True)

        initial_save_path = os.path.join(
            self.save_dir,
            f'{str(uid)}_initial.glb'
        )

        final_save_path = initial_save_path

        # -------------------------------------------------
        # 1. Shape generation
        # -------------------------------------------------
        try:
            self._load_shape_pipeline()

            shape_start = time.time()

            mesh = self.pipeline(image=image)[0]

            logger.info(
                "---Shape generation takes %.2f seconds ---",
                time.time() - shape_start
            )

            mesh.export(initial_save_path)

            logger.info(
                f"Initial mesh saved: {initial_save_path}"
            )

        except Exception as e:
            logger.error(
                f"Shape generation failed: {e}"
            )

            raise ValueError(
                f"Failed to generate 3D mesh: {str(e)}"
            )

        finally:
            # Mesh is a CPU-side trimesh result in the original
            # workflow. Keep the exported mesh file, but release
            # the Shape pipeline before loading Paint.
            self._release_shape_pipeline()

        # -------------------------------------------------
        # 2. Paint generation
        # -------------------------------------------------
        texture_enabled = params.get(
            'texture',
            True
        )

        if texture_enabled:
            output_mesh_path_obj = os.path.join(
                self.save_dir,
                f'{str(uid)}_texturing.obj'
            )

            try:
                self._load_paint_pipeline()

                paint_start = time.time()

                textured_path_obj = self.paint_pipeline(
                    mesh_path=initial_save_path,
                    image_path=image,
                    output_mesh_path=output_mesh_path_obj,
                    save_glb=False
                )

                logger.info(
                    "---Texture generation takes %.2f seconds ---",
                    time.time() - paint_start
                )

                logger.info(
                    f"output_mesh_path: {output_mesh_path_obj}, "
                    f"textured_path: {textured_path_obj}"
                )

                if not textured_path_obj:
                    raise RuntimeError(
                        "Paint pipeline returned an empty output path"
                    )

                if not os.path.isfile(textured_path_obj):
                    raise FileNotFoundError(
                        f"Paint output OBJ not found: {textured_path_obj}"
                    )

                logger.info(
                    "Converting textured OBJ to PBR GLB..."
                )

                glb_path_textured = os.path.join(
                    self.save_dir,
                    f'{str(uid)}_texturing.glb'
                )

                quick_convert_with_obj2gltf(
                    textured_path_obj,
                    glb_path_textured
                )

                if (
                    not os.path.isfile(glb_path_textured)
                    or os.path.getsize(glb_path_textured) == 0
                ):
                    raise RuntimeError(
                        "PBR GLB conversion failed: output file is empty"
                    )

                final_save_path = os.path.join(
                    self.save_dir,
                    f'{str(uid)}_textured.glb'
                )

                if os.path.exists(final_save_path):
                    os.remove(final_save_path)

                os.replace(
                    glb_path_textured,
                    final_save_path
                )

                logger.info(
                    f"Textured GLB generated: {final_save_path}"
                )

            except Exception as e:
                logger.error(
                    f"Texture generation failed: {e}"
                )

                # Preserve the original official fallback behavior.
                final_save_path = initial_save_path

                logger.warning(
                    f"Using untextured mesh as fallback: "
                    f"{final_save_path}"
                )

            finally:
                self._release_paint_pipeline()

        else:
            logger.info(
                "Texture generation disabled. Returning initial mesh."
            )

            final_save_path = initial_save_path

        # -------------------------------------------------
        # 3. Final cleanup
        # -------------------------------------------------
        if self.low_vram_mode:
            cuda_cleanup()

        logger.info(
            "---Total generation takes %.2f seconds ---",
            time.time() - start_time
        )

        return final_save_path, uid
