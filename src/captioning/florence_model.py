from __future__ import annotations

import importlib
import importlib.machinery
import logging
import sys
import types
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class FlorenceModel:
    """
    Caption de imágenes con Microsoft Florence-2.
    Incluye el parche de flash_attn encapsulado y documentado
    como deuda técnica a eliminar cuando flash_attn esté disponible.
    """
    def __init__(self, config):
        self.model_name: str = config.model_name
        self.task_prompt: str = config.task_prompt
        self.device: str = "cpu"
        self.dtype = torch.float32
        self._model = None
        self._processor = None

    def load(self) -> None:
        self._apply_flash_attn_patch()
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self._model.eval()
        logger.info(f"FlorenceModel cargado: {self.model_name}")

    def unload(self) -> None:
        self._model = None
        self._processor = None
        logger.info("FlorenceModel descargado")

    def caption(self, image: Image.Image, task_prompt: Optional[str] = None) -> str:
        if self._model is None:
            raise RuntimeError("Llama a load() antes de caption()")

        prompt = task_prompt or self.task_prompt
        inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self._processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height),
        )
        result = parsed.get(prompt, "")
        if isinstance(result, dict):
            result = result.get("caption", str(result))
        return str(result).strip()

    @staticmethod
    def _apply_flash_attn_patch() -> None:
        """
        DEUDA TÉCNICA: Florence-2 verifica flash_attn en su carga.
        Este parche registra un módulo mock en sys.modules para evitar
        el ImportError cuando flash_attn no está instalado (CPU / Windows).
        Eliminar cuando flash_attn esté disponible en el entorno.
        """
        class _MockModule(types.ModuleType):
            def __init__(self, name):
                super().__init__(name)
                self.__dict__["__spec__"] = importlib.machinery.ModuleSpec(name, None)
                self.__dict__["__version__"] = "2.0.0"
                self.__dict__["__file__"] = "<mock>"
                self.__dict__["__path__"] = []

            def __getattr__(self, name):
                if name in ("__path__", "__spec__"):
                    return []
                sub = _MockModule(f"{self.__name__}.{name}")
                sys.modules[sub.__name__] = sub
                return sub

        flash_attn_modules = [
            "flash_attn",
            "flash_attn.bert_padding",
            "flash_attn.flash_attn_interface",
            "flash_attn.flash_attn_func",
            "flash_attn.flash_attn_triton",
            "flash_attn.layers",
            "flash_attn.layers.rotary",
            "flash_attn.ops",
            "flash_attn.ops.fused_dense",
            "flash_attn.ops.layer_norm",
            "flash_attn.ops.rms_norm",
        ]
        for module_name in flash_attn_modules:
            if module_name not in sys.modules:
                sys.modules[module_name] = _MockModule(module_name)

        try:
            from transformers.utils import import_utils
            _orig = import_utils._is_package_available

            def _patched(pkg_name, return_version=False):
                if pkg_name == "flash_attn":
                    return (True, "2.0.0") if return_version else True
                return _orig(pkg_name, return_version)

            import_utils._is_package_available = _patched
            import_utils._flash_attn_2_available = True
        except Exception:
            pass

        logger.debug("Parche flash_attn aplicado")
