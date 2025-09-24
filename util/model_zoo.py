from util.constant import MODEL_ZOO
from model_zoo.LLaVA.utils import disable_torch_init


def get_pretrained_model(model_type: str):
    disable_torch_init()
    assert model_type in MODEL_ZOO.keys()
    model_path = MODEL_ZOO[model_type]
    kwargs = {}
    if model_type == "LLaVA":
        from model_zoo.LLaVA.model.builder import load_pretrained_model
        from model_zoo.LLaVA.mm_utils import get_model_name_from_path
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)
        model.config.tokenizer_padding_side = tokenizer.padding_side = "left"
        kwargs.update({'model_config': model.config,
                       'tokenizer': tokenizer,
                       'image_processor': image_processor,
                       'model': model})
    elif model_type == 'Molmo':
        from model_zoo.MolmoD import MolmoForCausalLM, MolmoProcessor
        model = MolmoForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        processor = MolmoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        tokenizer = processor.tokenizer
        kwargs.update({
            'model': model,
            'image_processor': processor,
            'tokenizer': tokenizer
        })
    elif model_type == 'Cambrian':
        from model_zoo.Cambrian.model.builder import load_pretrained_model
        from model_zoo.Cambrian.mm_utils import get_model_name_from_path
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
        model.config.tokenizer_padding_side = tokenizer.padding_side = "left"
        kwargs.update({'model_config': model.config,
                       'tokenizer': tokenizer,
                       'image_processor': image_processor,
                       'model': model})
    else:
        raise ValueError()
    return kwargs
