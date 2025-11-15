import yaml
from transformers import AutoTokenizer

CFG = yaml.safe_load(open("config.yaml","r"))

TOK = AutoTokenizer.from_pretrained(
    CFG.get("tokenizer_model_id", "stabilityai/stable-code-instruct-3b"),
    trust_remote_code=True
)

def _ntoks(text: str) -> int:
    return len(TOK.encode(text, add_special_tokens=False))

def build_prompt(question: str, context: str) -> str:
    sys_msg = CFG["prompt"]["system_message"]
    tmpl    = CFG["prompt"]["template"]
    sep     = CFG["prompt"].get("context_separator", "\n---\n")
    budget  = int(CFG["prompt"].get("max_input_tokens", 3800))

    blocks = [b for b in context.split(sep) if b.strip()]

    def fmt(ctx_blocks, q, s):
        ctx = sep.join(ctx_blocks)
        return tmpl.format(system_message=s, question=q, context=ctx)

    # Start with all blocks, remove the least-relevant (last) until under budget
    prompt = fmt(blocks, question, sys_msg)
    while _ntoks(prompt) > budget and blocks:
        blocks.pop()              # drop the last block
        prompt = fmt(blocks, question, sys_msg)

    # If still too long (very long question/system), trim question from the left
    if _ntoks(prompt) > budget:
        q_ids = TOK.encode(question, add_special_tokens=False)
        # keep the last 512 tokens of the question (adjust if needed)
        question = TOK.decode(q_ids[-512:], skip_special_tokens=False)
        prompt = fmt(blocks, question, sys_msg)

    # As a last resort, lightly trim the system message
    while _ntoks(prompt) > budget and len(sys_msg) > 400:
        sys_msg = sys_msg[: len(sys_msg) - 200]
        prompt = fmt(blocks, question, sys_msg)

    # Final clamp: if something is still off, hard truncate context tokens
    if _ntoks(prompt) > budget and blocks:
        # replace context with as many tokens as fit
        ctx_text = sep.join(blocks)
        ctx_ids = TOK.encode(ctx_text, add_special_tokens=False)
        keep = budget - _ntoks(tmpl.format(system_message=sys_msg, question=question, context=""))
        keep = max(keep, 0)
        ctx_trim = TOK.decode(ctx_ids[:keep], skip_special_tokens=False)
        prompt = tmpl.format(system_message=sys_msg, question=question, context=ctx_trim)

    return prompt
