"""
HuggingFace reference: dump intermediate values during prefill for TinyLlama-1.1B-Chat
so we can compare against Ksana's [DBG-PREFILL] / [DBG-MHA] / [DBG-EMB] logs.

Usage:
  python3 dbg_hf_reference.py --prompt "Hi" --layers 0,21
"""
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/zhichao.yang/KsanaLLM/src/ksana_llm/python/TinyLlama-1.1B-Chat")
    ap.add_argument("--prompt", default="Hi")
    ap.add_argument("--layers", default="all",
                    help="comma-separated layer indices to dump, or 'all'")
    ap.add_argument("--token", type=int, default=1,
                    help="token index to dump (default 1 = second token, matching Ksana DBG-PFL)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16,
             "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(args.device).eval()

    cfg = model.config
    if args.layers.strip() == "all":
        layers = list(range(cfg.num_hidden_layers))
    else:
        layers = sorted(int(x) for x in args.layers.split(","))
    print(f"=== model: {args.model}")
    print(f"=== config: hidden={cfg.hidden_size} heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} head_dim={cfg.hidden_size//cfg.num_attention_heads} layers={cfg.num_hidden_layers} vocab={cfg.vocab_size}")
    print(f"=== dtype={dtype} device={args.device}")

    enc = tok(args.prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc.input_ids.to(args.device)
    print(f"=== prompt: {args.prompt!r} -> ids = {input_ids.tolist()}")
    print(f"=== detokenized per id:")
    for i, t in enumerate(input_ids[0].tolist()):
        print(f"     [{i}] id={t} tok={tok.decode([t])!r}")

    # === Hook intermediates ===
    captures = {}

    def emb_hook(mod, inp, out):
        captures["emb"] = out.detach().clone()

    model.model.embed_tokens.register_forward_hook(emb_hook)

    layer_caps = {l: {} for l in layers}

    def make_layer_io_hook(layer_idx):
        def hook(mod, inp, out):
            x_in = inp[0] if isinstance(inp, tuple) else inp
            x_out = out[0] if isinstance(out, tuple) else out
            layer_caps[layer_idx]["resid_in"] = x_in.detach().clone()
            layer_caps[layer_idx]["resid_out"] = x_out.detach().clone()
        return hook

    def make_hooks(layer_idx, layer):
        def in_norm_hook(mod, inp, out):
            layer_caps[layer_idx]["pre_norm_out"] = out.detach().clone()
        def post_attn_norm_hook(mod, inp, out):
            layer_caps[layer_idx]["post_attn_norm_out"] = out.detach().clone()
        def qproj_hook(mod, inp, out):
            layer_caps[layer_idx]["q"] = out.detach().clone()
        def kproj_hook(mod, inp, out):
            layer_caps[layer_idx]["k"] = out.detach().clone()
        def vproj_hook(mod, inp, out):
            layer_caps[layer_idx]["v"] = out.detach().clone()
        def attn_out_hook(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            layer_caps[layer_idx]["attn_out"] = x.detach().clone()
        def mlp_out_hook(mod, inp, out):
            layer_caps[layer_idx]["mlp_out"] = out.detach().clone()

        layer.input_layernorm.register_forward_hook(in_norm_hook)
        layer.post_attention_layernorm.register_forward_hook(post_attn_norm_hook)
        layer.self_attn.q_proj.register_forward_hook(qproj_hook)
        layer.self_attn.k_proj.register_forward_hook(kproj_hook)
        layer.self_attn.v_proj.register_forward_hook(vproj_hook)
        layer.self_attn.register_forward_hook(attn_out_hook)
        layer.mlp.register_forward_hook(mlp_out_hook)

    for l in layers:
        make_hooks(l, model.model.layers[l])
        model.model.layers[l].register_forward_hook(make_layer_io_hook(l))

    # === Forward ===
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0, -1]  # last token logits

    print()
    print("=== EMBEDDING (per token, first 8 dims) ===")
    emb = captures["emb"][0]  # (T, H)
    for t, tid in enumerate(input_ids[0].tolist()):
        v = emb[t].float().cpu().numpy()
        print(f"  t={t} id={tid} emb[0..7]={v[:8].tolist()}")
        print(f"           emb_norm={float((emb[t].float()**2).sum().sqrt()):.4f}")

    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = cfg.hidden_size // nq

    # === Compact one-line-per-layer dump matching Ksana [DBG-PFL] format ===
    print()
    print(f"=== HF DBG-PFL t={args.token} ===")
    for l in layers:
        c = layer_caps[l]
        t = args.token
        pn = c["pre_norm_out"][0]
        attn = c["attn_out"][0]
        post_norm = c["post_attn_norm_out"][0]
        mlp = c["mlp_out"][0]
        if t >= pn.shape[0]:
            continue
        def f4(x):
            return [round(float(v), 4) for v in x[:4].float().cpu().tolist()]
        def nm(x):
            return float((x.float() ** 2).sum().sqrt())
        print(
            f"[HF-PFL L={l} t={t}] "
            f"pre_attn_norm[0..3]={f4(pn[t])} norm={nm(pn[t]):.3f} | "
            f"post_attn[0..3]={f4(attn[t])} norm={nm(attn[t]):.3f} | "
            f"pre_mlp_norm[0..3]={f4(post_norm[t])} norm={nm(post_norm[t]):.3f} | "
            f"post_mlp[0..3]={f4(mlp[t])} norm={nm(mlp[t]):.3f}"
        )
        if "resid_in" in c and "resid_out" in c:
            ri = c["resid_in"][0][t]
            ro = c["resid_out"][0][t]
            print(
                f"[HF-RES L={l} t={t}] "
                f"resid_in[0..3]={f4(ri)} norm={nm(ri):.3f} | "
                f"resid_out[0..3]={f4(ro)} norm={nm(ro):.3f}"
            )

    if len(layers) > 5:
        print(f"=== Skipping per-layer detailed dump (|layers|={len(layers)} > 5). Pass --layers 0,1 etc for detail. ===")
    for l in (layers if len(layers) <= 5 else []):
        c = layer_caps[l]
        print()
        print(f"=== LAYER {l} ===")
        pn = c["pre_norm_out"][0]  # (T, H)
        q = c["q"][0]               # (T, nq*hd)
        k = c["k"][0]               # (T, nkv*hd)
        v = c["v"][0]               # (T, nkv*hd)
        attn = c["attn_out"][0]     # (T, H)
        mlp = c["mlp_out"][0]       # (T, H)
        T = pn.shape[0]
        for t in range(T):
            qh0 = q[t, :hd].float().cpu().numpy()
            kh0 = k[t, :hd].float().cpu().numpy()
            vh0 = v[t, :hd].float().cpu().numpy()
            print(f"  t={t}  pre_norm_out[0..7]={pn[t,:8].float().cpu().tolist()}")
            print(f"         Q head0 [0..7]={qh0[:8].tolist()} ||Q_h0||={float((q[t,:hd].float()**2).sum().sqrt()):.4f}")
            print(f"         K head0 [0..7]={kh0[:8].tolist()} ||K_h0||={float((k[t,:hd].float()**2).sum().sqrt()):.4f}")
            print(f"         V head0 [0..7]={vh0[:8].tolist()} ||V_h0||={float((v[t,:hd].float()**2).sum().sqrt()):.4f}")
            print(f"         attn_out[0..7]={attn[t,:8].float().cpu().tolist()} norm={float((attn[t].float()**2).sum().sqrt()):.4f}")
            print(f"         mlp_out [0..7]={mlp[t,:8].float().cpu().tolist()} norm={float((mlp[t].float()**2).sum().sqrt()):.4f}")

    print()
    print("=== FINAL LOGITS (last token) ===")
    topk = logits.float().topk(10)
    for v, i in zip(topk.values.cpu().tolist(), topk.indices.cpu().tolist()):
        print(f"   id={i:6d}  logit={v:.4f}  tok={tok.decode([i])!r}")
    print(f"=== argmax = {int(logits.argmax())}  tok={tok.decode([int(logits.argmax())])!r}")


if __name__ == "__main__":
    main()
