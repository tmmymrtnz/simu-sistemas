#!/usr/bin/env python3
import subprocess, tempfile, pathlib, time, json, numpy as np, pandas as pd

JAR = pathlib.Path(__file__).resolve().parents[1]/"simulacion/cim.jar"

def run_props(props:dict)->float:
    with tempfile.NamedTemporaryFile("w+",suffix=".properties",delete=False) as tf:
        for k,v in props.items(): tf.write(f"{k}={v}\n"); tf.flush()
        t0=time.perf_counter()
        subprocess.run(["java","-jar",str(JAR),tf.name],
                       stdout=subprocess.DEVNULL,check=True)
        return (time.perf_counter()-t0)*1000

def main():
    L,rc,r=20,1,0.25
    rows=[]
    for N in [200,500,1000,2000]:
        for M in [5,8,12,16]:
            ms=run_props(dict(N=N,L=L,M=M,rc=rc,r=r,periodic=True,outputBase='bench'))
            rows.append((N,M,ms))
    df=pd.DataFrame(rows,columns=["N","M","ms"])
    print(df.pivot_table(index="N",columns="M",values="ms",aggfunc='first'))

if __name__=="__main__":
    main()
