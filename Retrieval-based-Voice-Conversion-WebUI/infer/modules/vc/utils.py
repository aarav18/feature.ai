import os

from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    from fairseq.data.dictionary import Dictionary
    import torch.serialization

    #torch.serialization.add_safe_class("fairseq.data.dictionary", "Dictionary")

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [config.hubert_path],
        arg_overrides={"data": config.hubert_base_dir}
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
