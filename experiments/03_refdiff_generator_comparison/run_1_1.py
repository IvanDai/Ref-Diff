from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path
import h5py, matplotlib.pyplot as plt, numpy as np, torch, yaml
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from refdiff import GaussianDiffusion
from refdiff_1_1 import ConditionalUNet1D, iq_to_time
from rid2026 import ImpairmentParameters, RFSignalGenerator, RML2018_MODULATIONS, SignalConfig, sample_channel, sample_parameters

DEFAULT={"signal":{"frame_length":1024,"samples_per_symbol":[8],"guard_symbols":32,"rrc_alpha":[.1,.4],"gmsk_bt":.3,"message_bandwidth":.2,"am_modulation_index":1.,"fm_deviation":.35},"impairments":{"esn0_db":list(range(-20,31,2)),"timing_offset":[0.,16.],"sigma_clk":.0001,"delay_spreads":[0.,.5,1.,2.],"max_channel_taps":32,"channel_paths":16,"enabled":{"awgn":True,"timing_offset":True,"symbol_rate_offset":True,"phase_offset":True,"carrier_offset":True,"multipath":True}}}
def pick(v,rng): return v[int(rng.integers(len(v)))] if isinstance(v,list) else v
def merge(a,b):
    out=dict(a)
    for k,v in b.items(): out[k]=merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
    return out
def output_dir():
    root=Path(__file__).with_name("outputs"); stamp=datetime.now().strftime("%Y%m%d_%H%M%S"); p=root/stamp; n=1
    while p.exists(): p=root/f"{stamp}_{n:02d}"; n+=1
    p.mkdir(parents=True); return p
def dataset_sample(path,mod,snr,rng):
    with h5py.File(path,"r") as h:
        mods=[x.decode() if isinstance(x,bytes) else str(x) for x in h["modulation"][:]]; snrs=np.asarray(h["esn0_db"][:],float)
        if mod not in mods: raise ValueError(f"modulation {mod!r} is not present in {path}")
        ix=np.flatnonzero(np.isclose(snrs,snr,atol=1e-6));
        if not len(ix): raise ValueError(f"Es/N0 {snr} dB is not present in {path}")
        c=(mods.index(mod),int(ix[0]),int(rng.integers(h["y_rx"].shape[2])))
        return h["y_rx"][c],h["x_ref"][c],{"source":"dataset","coordinate":list(c),"dataset":str(path.resolve())}
def generated_sample(cfg,mod,snr,seed,rng):
    cfg=cfg or {}; cfg={k:(v if v is not None else {}) for k,v in cfg.items()}; c=merge(DEFAULT,cfg); s=c["signal"]; i=c["impairments"]; ov=cfg.get("parameters",{}); sps=int(pick(s["samples_per_symbol"],rng)); alpha=float(pick(ov["rrc_alpha"],rng) if "rrc_alpha" in ov else rng.uniform(*s["rrc_alpha"]))
    signal=SignalConfig(int(s["frame_length"]),sps,int(s["guard_symbols"]),alpha,float(s["gmsk_bt"]),float(s["message_bandwidth"]),float(s["am_modulation_index"]),float(s["fm_deviation"]))
    sampled=sample_parameters(i,int(i["max_channel_taps"]),rng); names=("timing_offset","symbol_rate_offset","phase_offset","carrier_offset","delay_spread")
    vals={n:float(pick(ov[n],rng)) if n in ov else getattr(sampled,n) for n in names}; paths=int(i["channel_paths"]); taps=sample_channel(vals["delay_spread"],int(i["max_channel_taps"]),rng,path_count=paths)
    imp=ImpairmentParameters(esn0_db=snr,**vals,channel_taps=taps,channel_path_count=paths); r=RFSignalGenerator(signal).generate(mod,imp,seed=seed+1)
    return np.stack((r.y_rx.real,r.y_rx.imag)).astype("f4"),np.stack((r.x_ref.real,r.x_ref.imag)).astype("f4"),{"source":"generated","rrc_alpha":alpha,"samples_per_symbol":sps,"impairments":{n:float(getattr(imp,n)) for n in ("esn0_db",)+names}}
def load_model(path,cfg,device):
    ck=torch.load(path,map_location=device,weights_only=False); saved=ck.get("config",{}); m=cfg.get("model",saved.get("model",{})); d=cfg.get("diffusion",saved.get("diffusion",{})); model=ConditionalUNet1D(int(m.get("base_channels",32)),tuple(m.get("channel_multipliers",(1,2,4)))).to(device); model.load_state_dict(ck.get("ema_model",ck.get("model",ck))); return model.eval(),GaussianDiffusion(**d).to(device)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--weights",type=Path); p.add_argument("--dataset",type=Path); a=p.parse_args(); cfg=yaml.safe_load(a.config.read_text()) or {}; par=cfg.get("parameters",{})
    if "modulation" not in par or "esn0_db" not in par: raise ValueError("parameters.modulation and parameters.esn0_db are required")
    mod,snr=str(par["modulation"]),float(par["esn0_db"]); 
    if mod not in RML2018_MODULATIONS: raise ValueError(f"unsupported modulation: {mod!r}")
    seed=int(cfg.get("seed",233)); rng=np.random.default_rng(seed); ds=a.dataset or (Path(cfg["dataset"]) if cfg.get("dataset") else None)
    if ds: y,x,meta=dataset_sample(ds,mod,snr,rng)
    else: y,x,meta=generated_sample(cfg.get("generator",{}),mod,snr,seed,rng)
    dev=cfg.get("device","auto"); device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if dev=="auto" else dev); weights=Path(a.weights or cfg["weights"]); condition=iq_to_time(y).unsqueeze(0).to(device); target=iq_to_time(x); model,diff=load_model(weights,cfg,device); steps=min(int(cfg.get("ddim_steps",50)),diff.timesteps)
    with torch.no_grad(): restored=diff.ddim_sample(model,condition,steps=steps).cpu()[0]
    out=output_dir(); raw=condition.cpu()[0]; fig,ax=plt.subplots(3,2,figsize=(14,8),sharex=True)
    for row,(name,v) in enumerate((("y_rx",raw), ("x_ref",target), ("RefDiff output",restored))): ax[row,0].plot(v[0].numpy(),lw=.8); ax[row,0].set_ylabel(name); ax[row,0].set_title("real"); ax[row,1].plot(v[1].numpy(),lw=.8); ax[row,1].set_title("imag")
    ax[-1,0].set_xlabel("sample"); ax[-1,1].set_xlabel("sample"); fig.suptitle(f"{mod} | Es/N0={snr:g} dB | seed={seed}"); fig.tight_layout(); fig.savefig(out/"comparison.png",dpi=160); plt.close(fig)
    (out/"metadata.json").write_text(json.dumps({"model_version":"refdiff_1_1","domain":"time","weights":str(weights.resolve()),"modulation":mod,"esn0_db":snr,"seed":seed,"ddim_steps":steps,**meta},indent=2)); print(f"source={meta['source']}\noutput={out.resolve()}")
if __name__=="__main__": main()
