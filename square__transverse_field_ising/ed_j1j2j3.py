#!/usr/bin/env python

# coding:utf-8
from __future__ import print_function
import math
import numpy as np
#import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='transverse field Ising')
    parser.add_argument('-Lx',metavar='Lx',dest='Lx',type=np.int,default=2,help='set Lx')
    parser.add_argument('-Ly',metavar='Ly',dest='Ly',type=np.int,default=2,help='set Ly')
    parser.add_argument('-Jz1',metavar='Jz1',dest='Jz1',type=np.float64,default=-1.0,help='set Jz1 (default: Jz1=-1.0 (FM))')
    parser.add_argument('-Jz2',metavar='Jz2',dest='Jz2',type=np.float64,default=-0.0,help='set Jz2 (default: Jz2=-0.0 (FM))')
    parser.add_argument('-Jz3',metavar='Jz3',dest='Jz3',type=np.float64,default=-0.0,help='set Jz3 (default: Jz3=-0.0 (FM))')
    parser.add_argument('-Hz',metavar='Hz',dest='Hz',type=np.float64,default=0.0,help='set Hz')
    parser.add_argument('-Hx',metavar='Hx',dest='Hx',type=np.float64,default=1.0,help='set Hx')
    return parser.parse_args()

def make_spin():
    S0 = scipy.sparse.csr_matrix(np.array([[1,0],[0,1]]))
    Sx = scipy.sparse.csr_matrix(np.array([[0,1],[1,0]]))
    Sy = scipy.sparse.csr_matrix(np.array([[0,-1j],[1j,0]]))
    Sz = scipy.sparse.csr_matrix(np.array([[1,0],[0,-1]]))
    return S0, Sx, Sy, Sz

def make_Jz(Nbond,Jz):
    return [Jz for i in range(Nbond)]

def make_Hx(Ns,Hx):
    return [Hx for i in range(Ns)]

def make_Hz(Ns,Hz):
    return [Hz for i in range(Ns)]

def make_MF(Ns,Jz1,Jz2,Jz3,list_num_mf1,list_num_mf2,list_num_mf3):
    return [z1*Jz1 + z2*Jz2 + z3*Jz3 for z1,z2,z3 in zip(list_num_mf1,list_num_mf2,list_num_mf3)]

def make_Jz1_site(Lx,Ly):
    list_Jz1_site1 = []
    list_Jz1_site2 = []
    Nbond1 = 0
    for y in range(Ly):
        for x in range(Lx):
            if x<Lx-1:
                s1 = Lx*y+x
                s2 = Lx*y+(x+1)%Lx
                list_Jz1_site1.append(s1)
                list_Jz1_site2.append(s2)
                Nbond1 += 1
            if y<Ly-1:
                s1 = Lx*y+x
                s2 = Lx*((y+1)%Ly)+x
                list_Jz1_site1.append(s1)
                list_Jz1_site2.append(s2)
                Nbond1 += 1
    return list_Jz1_site1, list_Jz1_site2, Nbond1

def make_Jz2_site(Lx,Ly):
    list_Jz2_site1 = []
    list_Jz2_site2 = []
    Nbond2 = 0
    for y in range(Ly-1):
        for x in range(Lx-1):
            s1 = Lx*y+x
            s2 = Lx*((y+1)%Ly)+(x+1)%Lx
            list_Jz2_site1.append(s1)
            list_Jz2_site2.append(s2)
            Nbond2 += 1
            s1 = Lx*y+(x+1)%Lx
            s2 = Lx*((y+1)%Ly)+x
            list_Jz2_site1.append(s1)
            list_Jz2_site2.append(s2)
            Nbond2 += 1
    return list_Jz2_site1, list_Jz2_site2, Nbond2

def make_Jz3_site(Lx,Ly):
    list_Jz3_site1 = []
    list_Jz3_site2 = []
    Nbond3 = 0
    dist = 2
    for y in range(Ly):
        for x in range(Lx):
            if x<Lx-dist:
                s1 = Lx*y+x
                s2 = Lx*y+(x+dist)%Lx
                list_Jz3_site1.append(s1)
                list_Jz3_site2.append(s2)
                Nbond3 += 1
            if y<Ly-dist:
                s1 = Lx*y+x
                s2 = Lx*((y+dist)%Ly)+x
                list_Jz3_site1.append(s1)
                list_Jz3_site2.append(s2)
                Nbond3 += 1
    return list_Jz3_site1, list_Jz3_site2, Nbond3

def make_num_mf1(Lx,Ly):
    list_num_mf1 = []
    dist = 1
    for y in range(Ly):
        for x in range(Lx):
            cnt = 0
            if x+dist >= Lx:
                cnt += 1
            if x-dist <= -1:
                cnt += 1
            if y+dist >= Ly:
                cnt += 1
            if y-dist <= -1:
                cnt += 1
            list_num_mf1.append(cnt)
    return list_num_mf1

def make_num_mf2(Lx,Ly):
    list_num_mf2 = []
    dist = 1
    for y in range(Ly):
        for x in range(Lx):
            cnt = 0
            if x+dist >= Lx or y+dist >= Ly:
                cnt += 1
            if x+dist >= Lx or y-dist <= -1:
                cnt += 1
            if x-dist <= -1 or y+dist >= Ly:
                cnt += 1
            if x-dist <= -1 or y-dist <= -1:
                cnt += 1
            list_num_mf2.append(cnt)
    return list_num_mf2

def make_num_mf3(Lx,Ly):
    list_num_mf3 = []
    dist = 2
    for y in range(Ly):
        for x in range(Lx):
            cnt = 0
            if x+dist >= Lx:
                cnt += 1
            if x-dist <= -1:
                cnt += 1
            if y+dist >= Ly:
                cnt += 1
            if y-dist <= -1:
                cnt += 1
            list_num_mf3.append(cnt)
    return list_num_mf3

def make_Hx_site(Lx,Ly):
    list_Hx_site1 = [i for i in range(Lx*Ly)]
    return list_Hx_site1

def make_Hz_site(Lx,Ly):
    list_Hz_site1 = [i for i in range(Lx*Ly)]
    return list_Hz_site1

def make_hamiltonian(S0,Sx,Sy,Sz,Ns,\
    list_Jz1_site1,list_Jz1_site2,list_Jz2_site1,list_Jz2_site2,list_Jz3_site1,list_Jz3_site2,\
    list_Hz_site1,list_Hx_site1):
#
    list_SzSz_1 = []
    list_SzSz_2 = []
    list_SzSz_3 = []
    list_Sz = []
    list_Sx = []
#
    for i1,i2 in zip(list_Jz1_site1,list_Jz1_site2):
        SzSz = 1
        for site in range(Ns):
            if site==i1 or site==i2:
                SzSz = scipy.sparse.kron(SzSz,Sz,format='csr')
            else:
                SzSz = scipy.sparse.kron(SzSz,S0,format='csr')
        list_SzSz_1.append(SzSz)
#
    for i1,i2 in zip(list_Jz2_site1,list_Jz2_site2):
        SzSz = 1
        for site in range(Ns):
            if site==i1 or site==i2:
                SzSz = scipy.sparse.kron(SzSz,Sz,format='csr')
            else:
                SzSz = scipy.sparse.kron(SzSz,S0,format='csr')
        list_SzSz_2.append(SzSz)
#
    for i1,i2 in zip(list_Jz3_site1,list_Jz3_site2):
        SzSz = 1
        for site in range(Ns):
            if site==i1 or site==i2:
                SzSz = scipy.sparse.kron(SzSz,Sz,format='csr')
            else:
                SzSz = scipy.sparse.kron(SzSz,S0,format='csr')
        list_SzSz_3.append(SzSz)
#
    for i1 in list_Hz_site1:
        S0Sz = 1
        for site in range(Ns):
            if site==i1:
                S0Sz = scipy.sparse.kron(S0Sz,Sz,format='csr')
            else:
                S0Sz = scipy.sparse.kron(S0Sz,S0,format='csr')
        list_Sz.append(S0Sz)
#
    for i1 in list_Hx_site1:
        S0Sx = 1
        for site in range(Ns):
            if site==i1:
                S0Sx = scipy.sparse.kron(S0Sx,Sx,format='csr')
            else:
                S0Sx = scipy.sparse.kron(S0Sx,S0,format='csr')
        list_Sx.append(S0Sx)
#
    return list_SzSz_1, list_SzSz_2, list_SzSz_3, list_Sz, list_Sx

def make_sum_hamiltonian(list_SzSz_1,list_SzSz_2,list_SzSz_3,list_Sz,list_Sx,\
    list_Jz1,list_Jz2,list_Jz3,list_Hz,list_Hx,list_MF,mz0):
    Ham = \
          [val*opr for val,opr in zip(list_Jz1,list_SzSz_1)] \
        + [val*opr for val,opr in zip(list_Jz2,list_SzSz_2)] \
        + [val*opr for val,opr in zip(list_Jz3,list_SzSz_3)] \
        + [-val*opr for val,opr in zip(list_Hz,list_Sz)] \
        + [-val*opr for val,opr in zip(list_Hx,list_Sx)] \
        + [mz0*val*opr for val,opr in zip(list_MF,list_Sz)]
    Ham = np.sum(Ham)
    return Ham

def calc_mag(list_Sz,list_Sx,vec):
    list_mz = [vec.conjugate().dot(Sz.dot(vec)) for Sz in list_Sz]
    list_mx = [vec.conjugate().dot(Sx.dot(vec)) for Sx in list_Sx]
    mz = np.average(list_mz)
    mx = np.average(list_mx)
    return list_mz, list_mx, mz, mx

def self_consistent_loop(list_SzSz_1,list_SzSz_2,list_SzSz_3,list_Sz,list_Sx,\
    list_Jz1,list_Jz2,list_Jz3,list_Hz,list_Hx,list_MF):
    mz0 = 0.5 # initial guess
    mx0 = 0.0
    Nsteps = 1000
#    Nsteps = 100
    mageps = 1e-14
#    mageps = 1e-10
    for step in range(Nsteps):
        Ham = make_sum_hamiltonian(list_SzSz_1,list_SzSz_2,list_SzSz_3,list_Sz,list_Sx,\
            list_Jz1,list_Jz2,list_Jz3,list_Hz,list_Hx,list_MF,mz0)
        ene, vec = scipy.sparse.linalg.eigsh(Ham,k=1)
        list_mz, list_mx, mz, mx = calc_mag(list_Sz,list_Sx,vec[:,0])
        mzerr = np.abs(mz-mz0)
        mxerr = np.abs(mx-mx0)
#        print("# step,mz0,mz,mzerr",step,mz0,mz,mzerr)
        mz0 = mz
        mx0 = mx
        if mzerr < mageps and mxerr < mageps:
            break
    return step, mz, mzerr, mx, mxerr


def main():
    args = parse_args()
    Lx = args.Lx
    Ly = args.Ly
    Ns = Lx*Ly
    Jz1 = args.Jz1
    Jz2 = args.Jz2
    Jz3 = args.Jz3
    Hz = args.Hz
#    Hx = args.Hx
    print("# Lx",Lx)
    print("# Ly",Ly)
    print("# Ns",Ns)
    print("# Jz1",Jz1)
    print("# Jz2",Jz2)
    print("# Jz3",Jz3)
    print("# Hz",Hz)
#    print("# Hx",Hx)

    start = time.time()
    S0, Sx, Sy, Sz = make_spin()
    list_Jz1_site1, list_Jz1_site2, Nbond1 = make_Jz1_site(Lx,Ly)
    list_Jz1 = make_Jz(Nbond1,Jz1)
    print("# list_Jz1",list_Jz1)
    print("# list_Jz1_site1",list_Jz1_site1)
    print("# list_Jz1_site2",list_Jz1_site2)
    print("# Nbond1",Nbond1)
    list_Jz2_site1, list_Jz2_site2, Nbond2 = make_Jz2_site(Lx,Ly)
    list_Jz2 = make_Jz(Nbond2,Jz2)
    print("# list_Jz2",list_Jz2)
    print("# list_Jz2_site1",list_Jz2_site1)
    print("# list_Jz2_site2",list_Jz2_site2)
    print("# Nbond2",Nbond2)
    list_Jz3_site1, list_Jz3_site2, Nbond3 = make_Jz3_site(Lx,Ly)
    list_Jz3 = make_Jz(Nbond3,Jz3)
    print("# list_Jz3",list_Jz3)
    print("# list_Jz3_site1",list_Jz3_site1)
    print("# list_Jz3_site2",list_Jz3_site2)
    print("# Nbond3",Nbond3)
    list_Hx_site1 = make_Hx_site(Lx,Ly)
#    list_Hx = make_Hx(Ns,Hx)
#    print("# list_Hx",list_Hx)
    print("# list_Hx_site1",list_Hx_site1)
#
    list_Hz_site1 = make_Hz_site(Lx,Ly)
    list_Hz = make_Hz(Ns,Hz)
    print("# list_Hz",list_Hz)
    print("# list_Hz_site1",list_Hz_site1)
#
    list_num_mf1 = make_num_mf1(Lx,Ly)
    print("# list_num_mf1",list_num_mf1)
    list_num_mf2 = make_num_mf2(Lx,Ly)
    print("# list_num_mf2",list_num_mf2)
    list_num_mf3 = make_num_mf3(Lx,Ly)
    print("# list_num_mf3",list_num_mf3)
#
    list_MF = make_MF(Ns,Jz1,Jz2,Jz3,list_num_mf1,list_num_mf2,list_num_mf3)
    print("# list_MF",list_MF)
    end = time.time()
    print("## time: make interaction",end-start)

    start = time.time()
    list_SzSz_1, list_SzSz_2, list_SzSz_3, list_Sz, list_Sx = \
        make_hamiltonian(S0,Sx,Sy,Sz,Ns,\
        list_Jz1_site1,list_Jz1_site2,list_Jz2_site1,list_Jz2_site2,list_Jz3_site1,list_Jz3_site2,\
        list_Hz_site1,list_Hx_site1)
    end = time.time()
    print("## time: make each Hamiltonian",end-start)

    Hxs = np.linspace(0.0,14.0,71)

    for Hx in Hxs:
        start = time.time()
        list_Hx = make_Hx(Ns,Hx)
#        print("# list_Hx",list_Hx)
        step, mz, mzerr, mx, mxerr = \
            self_consistent_loop(list_SzSz_1,list_SzSz_2,list_SzSz_3,list_Sz,list_Sx,\
            list_Jz1,list_Jz2,list_Jz3,list_Hz,list_Hx,list_MF)
        print(Jz1,Jz2,Jz3,Hz,Hx,step,mz,mzerr,mx,mxerr)
        end = time.time()
        print("## time: self consistent loop",end-start)

if __name__ == "__main__":
    main()
