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
    parser.add_argument('-Jz',metavar='Jz',dest='Jz',type=np.float64,default=-1.0,help='set Jz (default: Jz=-1 (FM))')
    parser.add_argument('-Hz',metavar='Hz',dest='Hz',type=np.float64,default=0.0,help='set Hz')
#    parser.add_argument('-Hx',metavar='Hx',dest='Hx',type=np.float64,default=1.0,help='set Hx')
    return parser.parse_args()

def make_spin():
    S0 = scipy.sparse.csr_matrix(np.array([[1,0],[0,1]]))
    Sx = scipy.sparse.csr_matrix(np.array([[0,1],[1,0]]))
    Sy = scipy.sparse.csr_matrix(np.array([[0,-1j],[1j,0]]))
    Sz = scipy.sparse.csr_matrix(np.array([[1,0],[0,-1]]))
    return S0, Sx, Sy, Sz

def make_Jz_list(Lx,Ly):
    list_Jz_site1 = []
    list_Jz_site2 = []
    Ns = Lx*Ly
    Nbond = 0
    for y in range(Ly):
        for x in range(Lx):
            if x<Lx-1:
                s1 = Lx*y+x
                s2 = Lx*y+(x+1)%Lx
                list_Jz_site1.append(s1)
                list_Jz_site2.append(s2)
                Nbond += 1
            if y<Ly-1:
                s1 = Lx*y+x
                s2 = Lx*((y+1)%Ly)+x
                list_Jz_site1.append(s1)
                list_Jz_site2.append(s2)
                Nbond += 1
    return list_Jz_site1, list_Jz_site2, Ns, Nbond

def make_Hx_list(Lx,Ly):
    list_Hx_site1 = [i for i in range(Lx*Ly)]
    return list_Hx_site1

def make_Hz_list(Lx,Ly):
    list_Hz_site1 = [] # corners: 4 points --> Hz += -2*J*mz
    list_Hz_site2 = [] # edges: 2*(Lx-2)+2*(Ly-2) points --> Hz += -J*mz
    list_Hz_site3 = [] # others: (L-2)^2 points --> Hz += 0
    for y in [0,Ly-1]:
        for x in [0,Lx-1]:
            list_Hz_site1.append(Lx*y+x)
    for y in [0,Ly-1]:
        for x in range(1,Lx-1):
            list_Hz_site2.append(Lx*y+x)
    for y in range(1,Ly-1):
        for x in [0,Lx-1]:
            list_Hz_site2.append(Lx*y+x)
    for y in range(1,Ly-1):
        for x in range(1,Lx-1):
            list_Hz_site3.append(Lx*y+x)
    return list_Hz_site1, list_Hz_site2, list_Hz_site3

def make_hamiltonian(S0,Sx,Sy,Sz,Ns,list_Jz_site1,list_Jz_site2,\
    list_Hz_site1,list_Hz_site2,list_Hz_site3,list_Hx_site1):
#
    list_SzSz = []
    list_Sz = []
    list_Sz_1 = []
    list_Sz_2 = []
    list_Sz_3 = []
    list_Sx = []
#
    Nlist = len(list_Jz_site1)
    for ind in range(Nlist):
        i1 = list_Jz_site1[ind]
        i2 = list_Jz_site2[ind]
        SzSz = 1
        for site in range(Ns):
            if site==i1 or site==i2:
                SzSz = scipy.sparse.kron(SzSz,Sz,format='csr')
            else:
                SzSz = scipy.sparse.kron(SzSz,S0,format='csr')
        list_SzSz.append(SzSz)
#
    Nlist = len(list_Hz_site1) + len(list_Hz_site2) + len(list_Hz_site3)
    for ind in range(Nlist):
        S0Sz = 1
        for site in range(Ns):
            if site==ind:
                S0Sz = scipy.sparse.kron(S0Sz,Sz,format='csr')
            else:
                S0Sz = scipy.sparse.kron(S0Sz,S0,format='csr')
        list_Sz.append(S0Sz)
#
    Nlist = len(list_Hz_site1)
    for ind in range(Nlist):
        S0Sz = 1
        for site in range(Ns):
            if site==ind:
                S0Sz = scipy.sparse.kron(S0Sz,Sz,format='csr')
            else:
                S0Sz = scipy.sparse.kron(S0Sz,S0,format='csr')
        list_Sz_1.append(S0Sz)
#
    Nlist = len(list_Hz_site2)
    for ind in range(Nlist):
        S0Sz = 1
        for site in range(Ns):
            if site==ind:
                S0Sz = scipy.sparse.kron(S0Sz,Sz,format='csr')
            else:
                S0Sz = scipy.sparse.kron(S0Sz,S0,format='csr')
        list_Sz_2.append(S0Sz)
#
    Nlist = len(list_Hz_site3)
    for ind in range(Nlist):
        S0Sz = 1
        for site in range(Ns):
            if site==ind:
                S0Sz = scipy.sparse.kron(S0Sz,Sz,format='csr')
            else:
                S0Sz = scipy.sparse.kron(S0Sz,S0,format='csr')
        list_Sz_3.append(S0Sz)
#
    Nlist = len(list_Hx_site1)
    for ind in range(Nlist):
        S0Sx = 1
        for site in range(Ns):
            if site==ind:
                S0Sx = scipy.sparse.kron(S0Sx,Sx,format='csr')
            else:
                S0Sx = scipy.sparse.kron(S0Sx,S0,format='csr')
        list_Sx.append(S0Sx)
#
    return list_SzSz, list_Sz, list_Sz_1, list_Sz_2, list_Sz_3, list_Sx

def make_sum_hamiltonian(list_SzSz,list_Sz_1,list_Sz_2,list_Sz_3,list_Sx,\
    Jz,Hz_1,Hz_2,Hz_3,Hx):
    Ham = Jz * np.sum(list_SzSz) \
        - Hz_1 * np.sum(list_Sz_1) \
        - Hz_2 * np.sum(list_Sz_2) \
        - Hz_3 * np.sum(list_Sz_3) \
        - Hx * np.sum(list_Sx)
    return Ham

def calc_mag(list_Sz,list_Sx,vec):
    list_mz = [vec.conjugate().dot(Sz.dot(vec)) for Sz in list_Sz]
    list_mx = [vec.conjugate().dot(Sx.dot(vec)) for Sx in list_Sx]
    mz = np.average(list_mz)
    mx = np.average(list_mx)
    return list_mz, list_mx, mz, mx

def self_consistent_loop(list_SzSz,list_Sz,list_Sz_1,list_Sz_2,list_Sz_3,list_Sx,Jz,Hz,Hx):
    mz0 = 0.5 # initial guess
    mx0 = 0.0
    Hz0 = Hz - 2.0*Jz*mz0
    Hz1 = Hz - 1.0*Jz*mz0
    Hz2 = Hz
    Nsteps = 1000
#    Nsteps = 100
    mageps = 1e-14
#    mageps = 1e-10
    for step in range(Nsteps):
        Ham = make_sum_hamiltonian(list_SzSz,list_Sz_1,list_Sz_2,list_Sz_3,list_Sx,\
            Jz,Hz0,Hz1,Hz2,Hx)
        ene, vec = scipy.sparse.linalg.eigsh(Ham,k=1)
        list_mz, list_mx, mz, mx = calc_mag(list_Sz,list_Sx,vec[:,0])
        mzerr = np.abs(mz-mz0)
        mxerr = np.abs(mx-mx0)
#        print("# step,mz0,mz,mzerr",step,mz0,mz,mzerr)
        mz0 = mz
        Hz0 = Hz - 2.0*Jz*mz0
        Hz1 = Hz - 1.0*Jz*mz0
        Hz2 = Hz
        mx0 = mx
        if mzerr < mageps and mxerr < mageps:
            break
    return step, mz, mzerr, mx, mxerr


def main():
    args = parse_args()
    Lx = args.Lx
    Ly = args.Ly
    Jz = args.Jz
    Hz = args.Hz
#    Hx = args.Hx
    print("# Lx",Lx)
    print("# Ly",Ly)
    print("# Jz",Jz)
    print("# Hz",Hz)
#    print("# Hx",Hx)

    start = time.time()
    S0, Sx, Sy, Sz = make_spin()
    list_Jz_site1, list_Jz_site2, Ns, Nbond = make_Jz_list(Lx,Ly)
    print("# list_Jz_site1",list_Jz_site1)
    print("# list_Jz_site2",list_Jz_site2)
    print("# Ns",Ns)
    print("# Nbond",Nbond)
    list_Hx_site1 = make_Hx_list(Lx,Ly)
    print("# list_Hx_site1",list_Hx_site1)
    list_Hz_site1, list_Hz_site2, list_Hz_site3 = make_Hz_list(Lx,Ly)
    print("# list_Hz_site1",list_Hz_site1)
    print("# list_Hz_site2",list_Hz_site2)
    print("# list_Hz_site3",list_Hz_site3)
    end = time.time()
    print("## time: make interaction",end-start)

    start = time.time()
    list_SzSz, list_Sz, list_Sz_1, list_Sz_2, list_Sz_3, list_Sx = \
        make_hamiltonian(S0,Sx,Sy,Sz,Ns,list_Jz_site1,list_Jz_site2,\
        list_Hz_site1,list_Hz_site2,list_Hz_site3,list_Hx_site1)
    end = time.time()
    print("## time: make each Hamiltonian",end-start)

#    Hxs = np.linspace(0.0,4.0,41)
    Hxs = np.linspace(3.4,3.7,31)

    for Hx in Hxs:
        start = time.time()
        step, mz, mzerr, mx, mxerr = \
            self_consistent_loop(list_SzSz,list_Sz,list_Sz_1,list_Sz_2,list_Sz_3,list_Sx,Jz,Hz,Hx)
        print(Jz,Hz,Hx,step,mz,mzerr,mx,mxerr)
        end = time.time()
        print("## time: self consistent loop",end-start)

if __name__ == "__main__":
    main()
