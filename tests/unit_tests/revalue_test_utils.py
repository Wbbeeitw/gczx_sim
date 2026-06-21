import sys
import types


def install_omegaconf_stub() -> None:
    """Install the minimal OmegaConf stub needed by rlinf.__init__ in light tests."""
    if "omegaconf" in sys.modules:
        return

    omegaconf = types.ModuleType("omegaconf")

    class OmegaConf:
        @staticmethod
        def register_new_resolver(*args, **kwargs):
            return None

    omegaconf.OmegaConf = OmegaConf
    sys.modules["omegaconf"] = omegaconf
