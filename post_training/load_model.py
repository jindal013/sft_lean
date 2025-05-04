from vllm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_inference_model(model_name: str = "AI-MO/Kimina-Prover-Preview-Distill-1.5B") -> LLM:
    """
    Returns a vLLM LLM instance for fast batched inference.
    """
    llm = LLM(model=model_name, tensor_parallel_size=1)
    return llm


def get_training_model_and_tokenizer(model_name: str = "AI-MO/Kimina-Prover-Preview-Distill-1.5B"):
    """
    Returns a HuggingFace model and tokenizer for fine-tuning.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer